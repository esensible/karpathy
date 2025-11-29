// Package distribution provides types for representing and manipulating probability distributions.
package distribution

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// Distribution represents a probability distribution over values.
type Distribution interface {
	// Sample draws a random value from the distribution.
	Sample(rng *rand.Rand) Value
	// SampleN draws n random values from the distribution.
	SampleN(rng *rand.Rand, n int) []Value
	// Mean returns the expected value (for numeric distributions).
	Mean() (float64, bool)
	// Variance returns the variance (for numeric distributions).
	Variance() (float64, bool)
	// Support returns the interval bounds (for numeric distributions).
	Support() (float64, float64, bool)
	// Entropy returns the entropy of the distribution.
	Entropy() float64
	// Type returns the value type of this distribution.
	Type() ValueType
	// IsPoint returns true if this is a point distribution (known value).
	IsPoint() bool
	// PointValue returns the point value if IsPoint() is true.
	PointValue() (Value, bool)
	// Clone creates a deep copy of the distribution.
	Clone() Distribution
}

// ValueType represents the type of value a distribution produces.
type ValueType int

const (
	ValueTypeFloat ValueType = iota
	ValueTypeInt
	ValueTypeBool
	ValueTypeString
	ValueTypeEnum
	ValueTypeObject
)

// Value represents a concrete value that can be sampled from a distribution.
type Value struct {
	Type     ValueType
	Float    float64
	Int      int64
	Bool     bool
	String   string
	EnumType string // Name of the enum type
	Object   map[string]Value
}

// NewFloatValue creates a new float value.
func NewFloatValue(f float64) Value {
	return Value{Type: ValueTypeFloat, Float: f}
}

// NewIntValue creates a new int value.
func NewIntValue(i int64) Value {
	return Value{Type: ValueTypeInt, Int: i}
}

// NewBoolValue creates a new bool value.
func NewBoolValue(b bool) Value {
	return Value{Type: ValueTypeBool, Bool: b}
}

// NewStringValue creates a new string value.
func NewStringValue(s string) Value {
	return Value{Type: ValueTypeString, String: s}
}

// NewEnumValue creates a new enum value.
func NewEnumValue(enumType, value string) Value {
	return Value{Type: ValueTypeEnum, EnumType: enumType, String: value}
}

// NewObjectValue creates a new object value.
func NewObjectValue(obj map[string]Value) Value {
	return Value{Type: ValueTypeObject, Object: obj}
}

// AsFloat converts the value to float64 if possible.
func (v Value) AsFloat() (float64, bool) {
	switch v.Type {
	case ValueTypeFloat:
		return v.Float, true
	case ValueTypeInt:
		return float64(v.Int), true
	default:
		return 0, false
	}
}

// AsBool converts the value to bool if possible.
func (v Value) AsBool() (bool, bool) {
	if v.Type == ValueTypeBool {
		return v.Bool, true
	}
	return false, false
}

// AsString returns the string representation of the value.
func (v Value) AsString() string {
	switch v.Type {
	case ValueTypeFloat:
		return fmt.Sprintf("%g", v.Float)
	case ValueTypeInt:
		return fmt.Sprintf("%d", v.Int)
	case ValueTypeBool:
		return fmt.Sprintf("%t", v.Bool)
	case ValueTypeString, ValueTypeEnum:
		return v.String
	case ValueTypeObject:
		return fmt.Sprintf("%v", v.Object)
	default:
		return "<unknown>"
	}
}

// Equal returns true if the values are equal.
func (v Value) Equal(other Value) bool {
	if v.Type != other.Type {
		return false
	}
	switch v.Type {
	case ValueTypeFloat:
		return v.Float == other.Float
	case ValueTypeInt:
		return v.Int == other.Int
	case ValueTypeBool:
		return v.Bool == other.Bool
	case ValueTypeString, ValueTypeEnum:
		return v.String == other.String && v.EnumType == other.EnumType
	case ValueTypeObject:
		if len(v.Object) != len(other.Object) {
			return false
		}
		for k, val := range v.Object {
			otherVal, ok := other.Object[k]
			if !ok || !val.Equal(otherVal) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

// ToInterface converts Value to interface{} for use with expr-lang.
func (v Value) ToInterface() interface{} {
	switch v.Type {
	case ValueTypeFloat:
		return v.Float
	case ValueTypeInt:
		return v.Int
	case ValueTypeBool:
		return v.Bool
	case ValueTypeString:
		return v.String
	case ValueTypeEnum:
		return v.String
	case ValueTypeObject:
		result := make(map[string]interface{})
		for k, val := range v.Object {
			result[k] = val.ToInterface()
		}
		return result
	default:
		return nil
	}
}

// PointDistribution represents a known, fixed value.
type PointDistribution struct {
	value Value
}

// NewPointDistribution creates a new point distribution.
func NewPointDistribution(v Value) *PointDistribution {
	return &PointDistribution{value: v}
}

func (p *PointDistribution) Sample(rng *rand.Rand) Value    { return p.value }
func (p *PointDistribution) SampleN(rng *rand.Rand, n int) []Value {
	result := make([]Value, n)
	for i := range result {
		result[i] = p.value
	}
	return result
}
func (p *PointDistribution) Mean() (float64, bool) {
	return p.value.AsFloat()
}
func (p *PointDistribution) Variance() (float64, bool) {
	if _, ok := p.value.AsFloat(); ok {
		return 0, true
	}
	return 0, false
}
func (p *PointDistribution) Support() (float64, float64, bool) {
	if f, ok := p.value.AsFloat(); ok {
		return f, f, true
	}
	return 0, 0, false
}
func (p *PointDistribution) Entropy() float64     { return 0 }
func (p *PointDistribution) Type() ValueType      { return p.value.Type }
func (p *PointDistribution) IsPoint() bool        { return true }
func (p *PointDistribution) PointValue() (Value, bool) { return p.value, true }
func (p *PointDistribution) Clone() Distribution {
	return &PointDistribution{value: p.value}
}

// UniformDistribution represents a uniform distribution over [min, max].
type UniformDistribution struct {
	Min, Max float64
}

// NewUniformDistribution creates a new uniform distribution.
func NewUniformDistribution(min, max float64) *UniformDistribution {
	if min > max {
		min, max = max, min
	}
	return &UniformDistribution{Min: min, Max: max}
}

func (u *UniformDistribution) Sample(rng *rand.Rand) Value {
	return NewFloatValue(u.Min + rng.Float64()*(u.Max-u.Min))
}
func (u *UniformDistribution) SampleN(rng *rand.Rand, n int) []Value {
	result := make([]Value, n)
	for i := range result {
		result[i] = u.Sample(rng)
	}
	return result
}
func (u *UniformDistribution) Mean() (float64, bool) {
	return (u.Min + u.Max) / 2, true
}
func (u *UniformDistribution) Variance() (float64, bool) {
	d := u.Max - u.Min
	return d * d / 12, true
}
func (u *UniformDistribution) Support() (float64, float64, bool) {
	return u.Min, u.Max, true
}
func (u *UniformDistribution) Entropy() float64 {
	d := u.Max - u.Min
	if d <= 0 {
		return 0
	}
	return math.Log(d)
}
func (u *UniformDistribution) Type() ValueType { return ValueTypeFloat }
func (u *UniformDistribution) IsPoint() bool   { return u.Min == u.Max }
func (u *UniformDistribution) PointValue() (Value, bool) {
	if u.IsPoint() {
		return NewFloatValue(u.Min), true
	}
	return Value{}, false
}
func (u *UniformDistribution) Clone() Distribution {
	return &UniformDistribution{Min: u.Min, Max: u.Max}
}

// NormalDistribution represents a Gaussian distribution.
type NormalDistribution struct {
	Mu    float64
	Sigma float64
}

// NewNormalDistribution creates a new normal distribution.
func NewNormalDistribution(mu, sigma float64) *NormalDistribution {
	if sigma < 0 {
		sigma = -sigma
	}
	return &NormalDistribution{Mu: mu, Sigma: sigma}
}

func (n *NormalDistribution) Sample(rng *rand.Rand) Value {
	return NewFloatValue(rng.NormFloat64()*n.Sigma + n.Mu)
}
func (n *NormalDistribution) SampleN(rng *rand.Rand, count int) []Value {
	result := make([]Value, count)
	for i := range result {
		result[i] = n.Sample(rng)
	}
	return result
}
func (n *NormalDistribution) Mean() (float64, bool)    { return n.Mu, true }
func (n *NormalDistribution) Variance() (float64, bool) { return n.Sigma * n.Sigma, true }
func (n *NormalDistribution) Support() (float64, float64, bool) {
	// Use 6-sigma range as practical support
	return n.Mu - 6*n.Sigma, n.Mu + 6*n.Sigma, true
}
func (n *NormalDistribution) Entropy() float64 {
	return 0.5 * math.Log(2*math.Pi*math.E*n.Sigma*n.Sigma)
}
func (n *NormalDistribution) Type() ValueType { return ValueTypeFloat }
func (n *NormalDistribution) IsPoint() bool   { return n.Sigma == 0 }
func (n *NormalDistribution) PointValue() (Value, bool) {
	if n.IsPoint() {
		return NewFloatValue(n.Mu), true
	}
	return Value{}, false
}
func (n *NormalDistribution) Clone() Distribution {
	return &NormalDistribution{Mu: n.Mu, Sigma: n.Sigma}
}

// TruncatedNormalDistribution represents a truncated Gaussian.
type TruncatedNormalDistribution struct {
	Mu     float64
	Sigma  float64
	Lower  float64
	Upper  float64
	// Cached normalization constant
	normConst float64
}

// NewTruncatedNormalDistribution creates a new truncated normal distribution.
func NewTruncatedNormalDistribution(mu, sigma, lower, upper float64) *TruncatedNormalDistribution {
	if sigma < 0 {
		sigma = -sigma
	}
	if lower > upper {
		lower, upper = upper, lower
	}
	// Compute normalization constant
	alpha := (lower - mu) / sigma
	beta := (upper - mu) / sigma
	normConst := normalCDF(beta) - normalCDF(alpha)
	return &TruncatedNormalDistribution{
		Mu:        mu,
		Sigma:     sigma,
		Lower:     lower,
		Upper:     upper,
		normConst: normConst,
	}
}

func (t *TruncatedNormalDistribution) Sample(rng *rand.Rand) Value {
	// Rejection sampling
	for {
		x := rng.NormFloat64()*t.Sigma + t.Mu
		if x >= t.Lower && x <= t.Upper {
			return NewFloatValue(x)
		}
	}
}
func (t *TruncatedNormalDistribution) SampleN(rng *rand.Rand, n int) []Value {
	result := make([]Value, n)
	for i := range result {
		result[i] = t.Sample(rng)
	}
	return result
}
func (t *TruncatedNormalDistribution) Mean() (float64, bool) {
	// Approximate mean for truncated normal
	alpha := (t.Lower - t.Mu) / t.Sigma
	beta := (t.Upper - t.Mu) / t.Sigma
	if t.normConst == 0 {
		return (t.Lower + t.Upper) / 2, true
	}
	mean := t.Mu + t.Sigma*(normalPDF(alpha)-normalPDF(beta))/t.normConst
	return mean, true
}
func (t *TruncatedNormalDistribution) Variance() (float64, bool) {
	// Simplified variance approximation
	v := t.Sigma * t.Sigma
	// Reduce variance due to truncation
	return v * t.normConst, true
}
func (t *TruncatedNormalDistribution) Support() (float64, float64, bool) {
	return t.Lower, t.Upper, true
}
func (t *TruncatedNormalDistribution) Entropy() float64 {
	if t.normConst <= 0 {
		return 0
	}
	return math.Log(t.normConst * t.Sigma * math.Sqrt(2*math.Pi*math.E))
}
func (t *TruncatedNormalDistribution) Type() ValueType { return ValueTypeFloat }
func (t *TruncatedNormalDistribution) IsPoint() bool {
	return t.Lower == t.Upper || t.Sigma == 0
}
func (t *TruncatedNormalDistribution) PointValue() (Value, bool) {
	if t.IsPoint() {
		return NewFloatValue(t.Lower), true
	}
	return Value{}, false
}
func (t *TruncatedNormalDistribution) Clone() Distribution {
	return &TruncatedNormalDistribution{
		Mu:        t.Mu,
		Sigma:     t.Sigma,
		Lower:     t.Lower,
		Upper:     t.Upper,
		normConst: t.normConst,
	}
}

// BoolDistribution represents a Bernoulli distribution.
type BoolDistribution struct {
	ProbTrue float64
}

// NewBoolDistribution creates a new boolean distribution.
func NewBoolDistribution(pTrue float64) *BoolDistribution {
	if pTrue < 0 {
		pTrue = 0
	}
	if pTrue > 1 {
		pTrue = 1
	}
	return &BoolDistribution{ProbTrue: pTrue}
}

func (b *BoolDistribution) Sample(rng *rand.Rand) Value {
	return NewBoolValue(rng.Float64() < b.ProbTrue)
}
func (b *BoolDistribution) SampleN(rng *rand.Rand, n int) []Value {
	result := make([]Value, n)
	for i := range result {
		result[i] = b.Sample(rng)
	}
	return result
}
func (b *BoolDistribution) Mean() (float64, bool)    { return b.ProbTrue, true }
func (b *BoolDistribution) Variance() (float64, bool) { return b.ProbTrue * (1 - b.ProbTrue), true }
func (b *BoolDistribution) Support() (float64, float64, bool) {
	return 0, 1, true
}
func (b *BoolDistribution) Entropy() float64 {
	if b.ProbTrue <= 0 || b.ProbTrue >= 1 {
		return 0
	}
	return -b.ProbTrue*math.Log(b.ProbTrue) - (1-b.ProbTrue)*math.Log(1-b.ProbTrue)
}
func (b *BoolDistribution) Type() ValueType { return ValueTypeBool }
func (b *BoolDistribution) IsPoint() bool   { return b.ProbTrue == 0 || b.ProbTrue == 1 }
func (b *BoolDistribution) PointValue() (Value, bool) {
	if b.ProbTrue == 0 {
		return NewBoolValue(false), true
	}
	if b.ProbTrue == 1 {
		return NewBoolValue(true), true
	}
	return Value{}, false
}
func (b *BoolDistribution) Clone() Distribution {
	return &BoolDistribution{ProbTrue: b.ProbTrue}
}

// Confidence returns max(ProbTrue, 1-ProbTrue).
func (b *BoolDistribution) Confidence() float64 {
	if b.ProbTrue > 0.5 {
		return b.ProbTrue
	}
	return 1 - b.ProbTrue
}

// CategoricalDistribution represents a discrete distribution over string/enum values.
type CategoricalDistribution struct {
	Values  []string
	Probs   []float64
	EnumType string // Optional: name of the enum type
}

// NewCategoricalDistribution creates a new categorical distribution with uniform probabilities.
func NewCategoricalDistribution(values []string, enumType string) *CategoricalDistribution {
	n := len(values)
	probs := make([]float64, n)
	for i := range probs {
		probs[i] = 1.0 / float64(n)
	}
	return &CategoricalDistribution{Values: values, Probs: probs, EnumType: enumType}
}

// NewCategoricalDistributionWithProbs creates a categorical distribution with specified probabilities.
func NewCategoricalDistributionWithProbs(values []string, probs []float64, enumType string) *CategoricalDistribution {
	// Normalize probabilities
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	normalized := make([]float64, len(probs))
	for i, p := range probs {
		normalized[i] = p / sum
	}
	return &CategoricalDistribution{Values: values, Probs: normalized, EnumType: enumType}
}

func (c *CategoricalDistribution) Sample(rng *rand.Rand) Value {
	r := rng.Float64()
	cumProb := 0.0
	for i, p := range c.Probs {
		cumProb += p
		if r <= cumProb {
			if c.EnumType != "" {
				return NewEnumValue(c.EnumType, c.Values[i])
			}
			return NewStringValue(c.Values[i])
		}
	}
	// Fallback to last value
	if c.EnumType != "" {
		return NewEnumValue(c.EnumType, c.Values[len(c.Values)-1])
	}
	return NewStringValue(c.Values[len(c.Values)-1])
}
func (c *CategoricalDistribution) SampleN(rng *rand.Rand, n int) []Value {
	result := make([]Value, n)
	for i := range result {
		result[i] = c.Sample(rng)
	}
	return result
}
func (c *CategoricalDistribution) Mean() (float64, bool)    { return 0, false }
func (c *CategoricalDistribution) Variance() (float64, bool) { return 0, false }
func (c *CategoricalDistribution) Support() (float64, float64, bool) {
	return 0, 0, false
}
func (c *CategoricalDistribution) Entropy() float64 {
	h := 0.0
	for _, p := range c.Probs {
		if p > 0 {
			h -= p * math.Log(p)
		}
	}
	return h
}
func (c *CategoricalDistribution) Type() ValueType {
	if c.EnumType != "" {
		return ValueTypeEnum
	}
	return ValueTypeString
}
func (c *CategoricalDistribution) IsPoint() bool {
	nonZero := 0
	for _, p := range c.Probs {
		if p > 0 {
			nonZero++
		}
	}
	return nonZero <= 1
}
func (c *CategoricalDistribution) PointValue() (Value, bool) {
	for i, p := range c.Probs {
		if p == 1 {
			if c.EnumType != "" {
				return NewEnumValue(c.EnumType, c.Values[i]), true
			}
			return NewStringValue(c.Values[i]), true
		}
	}
	return Value{}, false
}
func (c *CategoricalDistribution) Clone() Distribution {
	values := make([]string, len(c.Values))
	copy(values, c.Values)
	probs := make([]float64, len(c.Probs))
	copy(probs, c.Probs)
	return &CategoricalDistribution{Values: values, Probs: probs, EnumType: c.EnumType}
}

// MostLikely returns the most likely value and its probability.
func (c *CategoricalDistribution) MostLikely() (string, float64) {
	maxProb := 0.0
	maxVal := ""
	for i, p := range c.Probs {
		if p > maxProb {
			maxProb = p
			maxVal = c.Values[i]
		}
	}
	return maxVal, maxProb
}

// EmpiricalDistribution represents a distribution estimated from samples.
type EmpiricalDistribution struct {
	Samples []Value
	sorted  []float64 // For numeric distributions
	vType   ValueType
}

// NewEmpiricalDistribution creates an empirical distribution from samples.
func NewEmpiricalDistribution(samples []Value) *EmpiricalDistribution {
	if len(samples) == 0 {
		return &EmpiricalDistribution{Samples: samples, vType: ValueTypeFloat}
	}
	vType := samples[0].Type

	// For numeric types, create sorted slice for percentile calculations
	var sorted []float64
	if vType == ValueTypeFloat || vType == ValueTypeInt {
		sorted = make([]float64, len(samples))
		for i, s := range samples {
			f, _ := s.AsFloat()
			sorted[i] = f
		}
		sort.Float64s(sorted)
	}

	return &EmpiricalDistribution{Samples: samples, sorted: sorted, vType: vType}
}

func (e *EmpiricalDistribution) Sample(rng *rand.Rand) Value {
	if len(e.Samples) == 0 {
		return Value{}
	}
	return e.Samples[rng.Intn(len(e.Samples))]
}
func (e *EmpiricalDistribution) SampleN(rng *rand.Rand, n int) []Value {
	result := make([]Value, n)
	for i := range result {
		result[i] = e.Sample(rng)
	}
	return result
}
func (e *EmpiricalDistribution) Mean() (float64, bool) {
	if len(e.sorted) == 0 {
		return 0, false
	}
	sum := 0.0
	for _, v := range e.sorted {
		sum += v
	}
	return sum / float64(len(e.sorted)), true
}
func (e *EmpiricalDistribution) Variance() (float64, bool) {
	mean, ok := e.Mean()
	if !ok {
		return 0, false
	}
	sum := 0.0
	for _, v := range e.sorted {
		d := v - mean
		sum += d * d
	}
	return sum / float64(len(e.sorted)), true
}
func (e *EmpiricalDistribution) Support() (float64, float64, bool) {
	if len(e.sorted) == 0 {
		return 0, 0, false
	}
	return e.sorted[0], e.sorted[len(e.sorted)-1], true
}
func (e *EmpiricalDistribution) Entropy() float64 {
	// Approximate entropy using histogram
	return math.Log(float64(len(e.Samples)))
}
func (e *EmpiricalDistribution) Type() ValueType { return e.vType }
func (e *EmpiricalDistribution) IsPoint() bool {
	if len(e.Samples) <= 1 {
		return true
	}
	// Check if all samples are the same
	first := e.Samples[0]
	for _, s := range e.Samples[1:] {
		if !s.Equal(first) {
			return false
		}
	}
	return true
}
func (e *EmpiricalDistribution) PointValue() (Value, bool) {
	if e.IsPoint() && len(e.Samples) > 0 {
		return e.Samples[0], true
	}
	return Value{}, false
}
func (e *EmpiricalDistribution) Clone() Distribution {
	samples := make([]Value, len(e.Samples))
	copy(samples, e.Samples)
	sorted := make([]float64, len(e.sorted))
	copy(sorted, e.sorted)
	return &EmpiricalDistribution{Samples: samples, sorted: sorted, vType: e.vType}
}

// Percentile returns the value at the given percentile (0-100).
func (e *EmpiricalDistribution) Percentile(p float64) (float64, bool) {
	if len(e.sorted) == 0 {
		return 0, false
	}
	if p <= 0 {
		return e.sorted[0], true
	}
	if p >= 100 {
		return e.sorted[len(e.sorted)-1], true
	}
	idx := int(p / 100.0 * float64(len(e.sorted)-1))
	return e.sorted[idx], true
}

// CredibleInterval returns the interval containing the given proportion of probability mass.
func (e *EmpiricalDistribution) CredibleInterval(alpha float64) (float64, float64, bool) {
	if len(e.sorted) == 0 {
		return 0, 0, false
	}
	lower := (1 - alpha) / 2 * 100
	upper := (1 - (1-alpha)/2) * 100
	l, _ := e.Percentile(lower)
	u, _ := e.Percentile(upper)
	return l, u, true
}

// Helper functions for normal distribution
func normalPDF(x float64) float64 {
	return math.Exp(-x*x/2) / math.Sqrt(2*math.Pi)
}

func normalCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}
