use std::borrow::Cow;
use std::hash::Hash;

use blobstore::Blob;
use blobstore::error::BlobstoreError;
use common::types::ScoreType;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError, ValidationErrors};

use crate::common::types::{DimId, DimOffset, DimWeight};

/// Sparse vector structure
#[derive(Debug, PartialEq, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct SparseVector {
    /// Indices must be unique
    pub indices: Vec<DimId>,
    /// Values and indices must be the same length
    pub values: Vec<DimWeight>,
}

impl Hash for SparseVector {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let Self { indices, values } = self;
        indices.hash(state);
        for &value in values {
            OrderedFloat(value).hash(state);
        }
    }
}

/// Same as `SparseVector` but with `DimOffset` indices.
/// Meaning that is uses internal segment-specific indices.
#[derive(Debug, PartialEq, Clone, Default, Serialize, Deserialize)]
pub struct RemappedSparseVector {
    /// indices must be unique
    pub indices: Vec<DimOffset>,
    /// values and indices must be the same length
    pub values: Vec<DimWeight>,
}

/// Sort two arrays by the first array.
pub fn double_sort<T: Ord + Copy, V: Copy>(indices: &mut [T], values: &mut [V]) {
    // Check if the indices are already sorted
    if indices.array_windows().all(|[a, b]| a < b) {
        return;
    }

    let mut indexed_values: Vec<(T, V)> = indices
        .iter()
        .zip(values.iter())
        .map(|(&i, &v)| (i, v))
        .collect();

    // Sort the vector of tuples by indices
    indexed_values.sort_unstable_by_key(|&(i, _)| i);

    for (i, (index, value)) in indexed_values.into_iter().enumerate() {
        indices[i] = index;
        values[i] = value;
    }
}

pub fn score_vectors<T: Ord + Eq>(
    self_indices: &[T],
    self_values: &[DimWeight],
    other_indices: &[T],
    other_values: &[DimWeight],
) -> Option<ScoreType> {
    let mut score = 0.0;
    // track whether there is any overlap
    let mut overlap = false;
    let mut i = 0;
    let mut j = 0;
    while i < self_indices.len() && j < other_indices.len() {
        match self_indices[i].cmp(&other_indices[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                overlap = true;
                score += self_values[i] * other_values[j];
                i += 1;
                j += 1;
            }
        }
    }
    overlap.then_some(score)
}

impl RemappedSparseVector {
    pub fn new(indices: Vec<DimId>, values: Vec<DimWeight>) -> Result<Self, ValidationErrors> {
        let vector = Self { indices, values };
        vector.validate()?;
        Ok(vector)
    }

    pub fn sort_by_indices(&mut self) {
        double_sort(&mut self.indices, &mut self.values);
    }

    /// Check if this vector is sorted by indices.
    pub fn is_sorted(&self) -> bool {
        self.indices.array_windows().all(|[a, b]| a < b)
    }

    /// Score this vector against another vector using dot product.
    /// Warning: Expects both vectors to be sorted by indices.
    ///
    /// Return None if the vectors do not overlap.
    pub fn score(&self, other: &RemappedSparseVector) -> Option<ScoreType> {
        debug_assert!(self.is_sorted());
        debug_assert!(other.is_sorted());
        score_vectors(&self.indices, &self.values, &other.indices, &other.values)
    }

    /// Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cut the vector to the dimensions with the largest absolute weights that together make up
    /// at least `weight_to_keep` of the total absolute weight. The vector is then re-sorted by
    /// indices.
    ///
    /// At least one dimension is always kept. The sums are computed in `f64`, so a fraction that
    /// lands exactly on a boundary depends on how `weight_to_keep` rounds in `f32`.
    pub fn cut(&mut self, weight_to_keep: f32) {
        // weight_to_keep should be already validated by the API
        // spec to make sure it is in (0, 1]
        debug_assert!(weight_to_keep > 0.0 && weight_to_keep <= 1.0);
        if weight_to_keep >= 1.0 {
            return;
        }
        let total_weight: f64 = self.values.iter().map(|v| f64::from(v.abs())).sum();
        if total_weight == 0.0 {
            return;
        }
        let target_weight = f64::from(weight_to_keep) * total_weight;

        // rank by (abs weight desc, index asc), so that ties are broken the same way every time
        let mut ranked = (0..self.len()).collect::<Vec<usize>>();
        ranked.sort_unstable_by(|&a, &b| {
            self.values[b]
                .abs()
                .total_cmp(&self.values[a].abs())
                .then_with(|| self.indices[a].cmp(&self.indices[b]))
        });

        // keep dimensions up to and including the one that reaches the target weight
        let mut kept_weight = 0.0;
        let keep = ranked
            .iter()
            .take_while(|&&i| {
                let reached = kept_weight >= target_weight;
                kept_weight += f64::from(self.values[i].abs());
                !reached
            })
            .count()
            .max(1);
        ranked.truncate(keep);

        self.indices = ranked.iter().map(|&i| self.indices[i]).collect();
        self.values = ranked.iter().map(|&i| self.values[i]).collect();
        self.sort_by_indices();
    }
}

impl SparseVector {
    pub fn new(indices: Vec<DimId>, values: Vec<DimWeight>) -> Result<Self, ValidationErrors> {
        let vector = SparseVector { indices, values };
        vector.validate()?;
        Ok(vector)
    }

    #[cfg(feature = "testing")]
    pub fn new_unchecked(indices: Vec<DimId>, values: Vec<DimWeight>) -> Self {
        SparseVector { indices, values }
    }

    /// Sort this vector by indices.
    ///
    /// Sorting is required for scoring and overlap checks.
    pub fn sort_by_indices(&mut self) {
        double_sort(&mut self.indices, &mut self.values);
    }

    /// Check if this vector is sorted by indices.
    pub fn is_sorted(&self) -> bool {
        self.indices.windows(2).all(|w| w[0] < w[1])
    }

    /// Check if this vector is empty.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() && self.values.is_empty()
    }

    /// Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Score this vector against another vector using dot product.
    /// Warning: Expects both vectors to be sorted by indices.
    ///
    /// Return None if the vectors do not overlap.
    pub fn score(&self, other: &SparseVector) -> Option<ScoreType> {
        debug_assert!(self.is_sorted());
        debug_assert!(other.is_sorted());
        score_vectors(&self.indices, &self.values, &other.indices, &other.values)
    }

    /// Construct a new vector that is the result of performing all indices-wise operations.
    /// Automatically sort input vectors if necessary.
    pub fn combine_aggregate(
        &self,
        other: &SparseVector,
        op: impl Fn(DimWeight, DimWeight) -> DimWeight,
    ) -> Self {
        // Copy and sort `self` vector if not already sorted
        let this: Cow<SparseVector> = if !self.is_sorted() {
            let mut this = self.clone();
            this.sort_by_indices();
            Cow::Owned(this)
        } else {
            Cow::Borrowed(self)
        };
        assert!(this.is_sorted());

        // Copy and sort `other` vector if not already sorted
        let cow_other: Cow<SparseVector> = if !other.is_sorted() {
            let mut other = other.clone();
            other.sort_by_indices();
            Cow::Owned(other)
        } else {
            Cow::Borrowed(other)
        };
        let other = &cow_other;
        assert!(other.is_sorted());

        let mut result = SparseVector::default();
        let mut i = 0;
        let mut j = 0;
        while i < this.indices.len() && j < other.indices.len() {
            match this.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Less => {
                    result.indices.push(this.indices[i]);
                    result.values.push(op(this.values[i], 0.0));
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.indices.push(other.indices[j]);
                    result.values.push(op(0.0, other.values[j]));
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    result.indices.push(this.indices[i]);
                    result.values.push(op(this.values[i], other.values[j]));
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < this.indices.len() {
            result.indices.push(this.indices[i]);
            result.values.push(op(this.values[i], 0.0));
            i += 1;
        }
        while j < other.indices.len() {
            result.indices.push(other.indices[j]);
            result.values.push(op(0.0, other.values[j]));
            j += 1;
        }
        debug_assert!(result.is_sorted());
        debug_assert!(result.validate().is_ok());
        result
    }

    /// Create [RemappedSparseVector] from this vector in a naive way. Only suitable for testing.
    #[cfg(feature = "testing")]
    pub fn into_remapped(self) -> RemappedSparseVector {
        RemappedSparseVector {
            indices: self.indices,
            values: self.values,
        }
    }
}

impl TryFrom<Vec<(u32, f32)>> for RemappedSparseVector {
    type Error = ValidationErrors;

    fn try_from(tuples: Vec<(u32, f32)>) -> Result<Self, Self::Error> {
        let (indices, values): (Vec<_>, Vec<_>) = tuples.into_iter().unzip();
        RemappedSparseVector::new(indices, values)
    }
}

impl TryFrom<Vec<(u32, f32)>> for SparseVector {
    type Error = ValidationErrors;

    fn try_from(tuples: Vec<(u32, f32)>) -> Result<Self, Self::Error> {
        let (indices, values): (Vec<_>, Vec<_>) = tuples.into_iter().unzip();
        SparseVector::new(indices, values)
    }
}

impl Blob for SparseVector {
    fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(&self).expect("Sparse vector serialization should not fail")
    }

    fn from_bytes(data: &[u8]) -> Result<Self, BlobstoreError> {
        bincode::deserialize(data).map_err(|err| {
            BlobstoreError::decode_error(format!("Failed to deserialize sparse vector: {err}"))
        })
    }
}

#[cfg(test)]
impl<const N: usize> From<[(u32, f32); N]> for SparseVector {
    fn from(value: [(u32, f32); N]) -> Self {
        value.to_vec().try_into().unwrap()
    }
}

#[cfg(test)]
impl<const N: usize> From<[(u32, f32); N]> for RemappedSparseVector {
    fn from(value: [(u32, f32); N]) -> Self {
        value.to_vec().try_into().unwrap()
    }
}

impl Validate for SparseVector {
    fn validate(&self) -> Result<(), ValidationErrors> {
        validate_sparse_vector_impl(&self.indices, &self.values)
    }
}

impl Validate for RemappedSparseVector {
    fn validate(&self) -> Result<(), ValidationErrors> {
        validate_sparse_vector_impl(&self.indices, &self.values)
    }
}

pub fn validate_sparse_vector_impl<T: Clone + Eq + Hash>(
    indices: &[T],
    values: &[DimWeight],
) -> Result<(), ValidationErrors> {
    let mut errors = ValidationErrors::default();

    if indices.len() != values.len() {
        errors.add(
            "values",
            ValidationError::new("must be the same length as indices"),
        );
    }
    if indices.iter().unique().count() != indices.len() {
        errors.add("indices", ValidationError::new("must be unique"));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    #[test]
    fn test_score_aligned_same_size() {
        let v1 = RemappedSparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let v2 = RemappedSparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(v1.score(&v2), Some(14.0));
    }

    #[test]
    fn test_score_not_aligned_same_size() {
        let v1 = RemappedSparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let v2 = RemappedSparseVector::new(vec![2, 3, 4], vec![2.0, 3.0, 4.0]).unwrap();
        assert_eq!(v1.score(&v2), Some(13.0));
    }

    #[test]
    fn test_score_aligned_different_size() {
        let v1 = RemappedSparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let v2 = RemappedSparseVector::new(vec![1, 2, 3, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(v1.score(&v2), Some(14.0));
    }

    #[test]
    fn test_score_not_aligned_different_size() {
        let v1 = RemappedSparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let v2 = RemappedSparseVector::new(vec![2, 3, 4, 5], vec![2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(v1.score(&v2), Some(13.0));
    }

    #[test]
    fn test_score_no_overlap() {
        let v1 = RemappedSparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let v2 = RemappedSparseVector::new(vec![4, 5, 6], vec![2.0, 3.0, 4.0]).unwrap();
        assert!(v1.score(&v2).is_none());
    }

    #[test]
    fn validation_test() {
        let fully_empty = SparseVector::new(vec![], vec![]);
        assert!(fully_empty.is_ok());
        assert!(fully_empty.unwrap().is_empty());

        let different_length = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0]);
        assert!(different_length.is_err());

        let not_sorted = SparseVector::new(vec![1, 3, 2], vec![1.0, 2.0, 3.0]);
        assert!(not_sorted.is_ok());

        let not_unique = SparseVector::new(vec![1, 2, 3, 2], vec![1.0, 2.0, 3.0, 4.0]);
        assert!(not_unique.is_err());
    }

    #[test]
    fn sorting_test() {
        let mut not_sorted = SparseVector::new(vec![1, 3, 2], vec![1.0, 2.0, 3.0]).unwrap();
        assert!(!not_sorted.is_sorted());
        not_sorted.sort_by_indices();
        assert!(not_sorted.is_sorted());
    }

    #[test]
    fn combine_aggregate_test() {
        // Test with missing index
        let a = SparseVector::new(vec![1, 2, 3], vec![0.1, 0.2, 0.3]).unwrap();
        let b = SparseVector::new(vec![2, 3, 4], vec![2.0, 3.0, 4.0]).unwrap();
        let sum = a.combine_aggregate(&b, |x, y| x + 2.0 * y);
        assert_eq!(sum.indices, vec![1, 2, 3, 4]);
        assert_eq!(sum.values, vec![0.1, 4.2, 6.3, 8.0]);

        // reverse arguments
        let sum = b.combine_aggregate(&a, |x, y| x + 2.0 * y);
        assert_eq!(sum.indices, vec![1, 2, 3, 4]);
        assert_eq!(sum.values, vec![0.2, 2.4, 3.6, 4.0]);

        // Test with non-sorted input
        let a = SparseVector::new(vec![1, 2, 3], vec![0.1, 0.2, 0.3]).unwrap();
        let b = SparseVector::new(vec![4, 2, 3], vec![4.0, 2.0, 3.0]).unwrap();
        let sum = a.combine_aggregate(&b, |x, y| x + 2.0 * y);
        assert_eq!(sum.indices, vec![1, 2, 3, 4]);
        assert_eq!(sum.values, vec![0.1, 4.2, 6.3, 8.0]);
    }

    /// Hand-computed cases. The fractions are exact in `f32`, and the weights are small integers,
    /// so no case lands on a rounding boundary by accident.
    #[test]
    fn cut_keeps_weight_fraction() {
        // by index: -2, 4, 1, 3. By absolute weight: 4, 3, 2, 1. The total is 10.
        let vector =
            RemappedSparseVector::new(vec![0, 1, 2, 3], vec![-2.0, 4.0, 1.0, 3.0]).unwrap();

        for (weight_to_keep, expected_indices) in [
            (0.0001, vec![1]),        // at least one dimension is kept
            (0.25, vec![1]),          // 4 >= 2.5
            (0.5, vec![1, 3]),        // 4 < 5, 4 + 3 = 7 >= 5
            (0.75, vec![0, 1, 3]),    // 7 < 7.5, 7 + 2 = 9 >= 7.5
            (0.95, vec![0, 1, 2, 3]), // 9 < 9.5, 9 + 1 = 10 >= 9.5
            (1.0, vec![0, 1, 2, 3]),  // everything
        ] {
            let mut cut = vector.clone();
            cut.cut(weight_to_keep);
            assert_eq!(
                cut.indices, expected_indices,
                "weight_to_keep = {weight_to_keep}"
            );
            let expected_values: Vec<DimWeight> = expected_indices
                .iter()
                .map(|&index| {
                    vector.values[vector.indices.iter().position(|&i| i == index).unwrap()]
                })
                .collect();
            assert_eq!(
                cut.values, expected_values,
                "weight_to_keep = {weight_to_keep}"
            );
        }
    }

    /// Dimensions with the same absolute weight are kept in index order, and a weight that is
    /// exactly reached keeps no further dimension.
    #[test]
    fn cut_breaks_ties_by_index() {
        let vector =
            RemappedSparseVector::new(vec![0, 1, 2, 3], vec![1.0, -1.0, 1.0, -1.0]).unwrap();

        let mut cut = vector.clone();
        cut.cut(0.5);
        assert_eq!(cut.indices, vec![0, 1]);
        assert_eq!(cut.values, vec![1.0, -1.0]);

        let mut cut = vector.clone();
        cut.cut(0.75);
        assert_eq!(cut.indices, vec![0, 1, 2]);
    }

    /// A vector without weight has nothing to rank, so it stays as it is.
    #[test]
    fn cut_leaves_zero_weight_vector() {
        let vector = RemappedSparseVector::new(vec![1, 4], vec![0.0, 0.0]).unwrap();
        let mut cut = vector.clone();
        cut.cut(0.5);
        assert_eq!(cut, vector);
    }

    /// Compare `cut` against a naive reference on random vectors, and check the defining
    /// properties of the result.
    ///
    /// Weights are drawn from a small pool of magnitudes with random signs, so that ties
    /// (including `+x` vs `-x`) are frequent, which the fixture's continuous values never produce.
    /// The magnitudes are multiples of 0.5, so all sums are exact in `f64`.
    #[test]
    fn cut_matches_reference() {
        const MAGNITUDES: [f32; 6] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        const FRACTIONS: [f32; 8] = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0];

        let mut rnd_gen = StdRng::seed_from_u64(42);

        for _ in 0..500 {
            let mut indices: Vec<DimOffset> =
                (0..100).filter(|_| rnd_gen.random_bool(0.3)).collect();
            if indices.is_empty() {
                indices.push(rnd_gen.random_range(0..100));
            }
            let values: Vec<DimWeight> = indices
                .iter()
                .map(|_| {
                    let magnitude = MAGNITUDES[rnd_gen.random_range(0..MAGNITUDES.len())];
                    if rnd_gen.random_bool(0.5) {
                        magnitude
                    } else {
                        -magnitude
                    }
                })
                .collect();
            let original = RemappedSparseVector::new(indices, values).unwrap();

            // fixed fractions cover the boundaries, random ones the rest of (0, 1]
            let weight_to_keep = if rnd_gen.random_bool(0.5) {
                FRACTIONS[rnd_gen.random_range(0..FRACTIONS.len())]
            } else {
                rnd_gen.random_range(f32::EPSILON..=1.0)
            };

            // reference: rank by (abs weight desc, index asc), keep ranks until the weight is
            // reached, restore index order
            let mut ranked: Vec<(DimOffset, DimWeight)> = original
                .indices
                .iter()
                .copied()
                .zip(original.values.iter().copied())
                .collect();
            ranked.sort_by(|&(i1, v1), &(i2, v2)| {
                v2.abs().total_cmp(&v1.abs()).then_with(|| i1.cmp(&i2))
            });
            let total: f64 = ranked.iter().map(|&(_, v)| f64::from(v.abs())).sum();
            let target = f64::from(weight_to_keep) * total;
            let mut kept_weight = 0.0;
            let mut expected = Vec::new();
            for &(index, value) in &ranked {
                if kept_weight >= target && !expected.is_empty() {
                    break;
                }
                kept_weight += f64::from(value.abs());
                expected.push((index, value));
            }
            expected.sort_by_key(|&(index, _)| index);
            let (expected_indices, expected_values): (Vec<_>, Vec<_>) =
                expected.into_iter().unzip();

            let mut cut = original.clone();
            cut.cut(weight_to_keep);

            let context = format!("weight_to_keep = {weight_to_keep}, input = {original:?}");
            assert!(cut.is_sorted(), "cut result must stay sorted by index");
            assert_eq!(cut.indices, expected_indices, "{context}");
            assert_eq!(cut.values, expected_values, "{context}");

            // the defining properties, independent of the reference
            let weights: Vec<f64> = cut.values.iter().map(|v| f64::from(v.abs())).collect();
            let result_weight: f64 = weights.iter().sum();
            let smallest_kept = weights.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(result_weight >= target, "too little weight kept: {context}");
            if cut.len() > 1 {
                assert!(
                    result_weight - smallest_kept < target,
                    "a dimension could have been dropped: {context}"
                );
            }
            let largest_dropped = original
                .indices
                .iter()
                .zip(&original.values)
                .filter(|(index, _)| !cut.indices.contains(index))
                .map(|(_, value)| f64::from(value.abs()))
                .fold(0.0, f64::max);
            assert!(
                smallest_kept >= largest_dropped,
                "a heavier dimension was dropped: {context}"
            );
        }
    }
}
