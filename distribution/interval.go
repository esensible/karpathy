package distribution

import (
	"math"
)

// Interval represents a numeric interval [Lower, Upper].
type Interval struct {
	Lower float64
	Upper float64
}

// NewInterval creates a new interval.
func NewInterval(lower, upper float64) Interval {
	if lower > upper {
		lower, upper = upper, lower
	}
	return Interval{Lower: lower, Upper: upper}
}

// Point creates a point interval [v, v].
func Point(v float64) Interval {
	return Interval{Lower: v, Upper: v}
}

// Entire returns the interval (-inf, +inf).
func Entire() Interval {
	return Interval{Lower: math.Inf(-1), Upper: math.Inf(1)}
}

// Empty returns an empty interval.
func Empty() Interval {
	return Interval{Lower: math.NaN(), Upper: math.NaN()}
}

// IsPoint returns true if the interval is a single point.
func (i Interval) IsPoint() bool {
	return i.Lower == i.Upper
}

// IsEmpty returns true if the interval is empty.
func (i Interval) IsEmpty() bool {
	return math.IsNaN(i.Lower) || math.IsNaN(i.Upper)
}

// Width returns the width of the interval.
func (i Interval) Width() float64 {
	if i.IsEmpty() {
		return 0
	}
	return i.Upper - i.Lower
}

// Midpoint returns the midpoint of the interval.
func (i Interval) Midpoint() float64 {
	if i.IsEmpty() {
		return math.NaN()
	}
	return (i.Lower + i.Upper) / 2
}

// Contains returns true if x is within the interval.
func (i Interval) Contains(x float64) bool {
	return x >= i.Lower && x <= i.Upper
}

// Overlaps returns true if the intervals overlap.
func (i Interval) Overlaps(other Interval) bool {
	return i.Lower <= other.Upper && other.Lower <= i.Upper
}

// Add returns the interval sum.
func (i Interval) Add(other Interval) Interval {
	return Interval{
		Lower: i.Lower + other.Lower,
		Upper: i.Upper + other.Upper,
	}
}

// Sub returns the interval difference.
func (i Interval) Sub(other Interval) Interval {
	return Interval{
		Lower: i.Lower - other.Upper,
		Upper: i.Upper - other.Lower,
	}
}

// Mul returns the interval product.
func (i Interval) Mul(other Interval) Interval {
	products := []float64{
		i.Lower * other.Lower,
		i.Lower * other.Upper,
		i.Upper * other.Lower,
		i.Upper * other.Upper,
	}
	return Interval{
		Lower: minFloat64(products...),
		Upper: maxFloat64(products...),
	}
}

// Div returns the interval quotient.
func (i Interval) Div(other Interval) Interval {
	// Handle division by zero
	if other.Lower <= 0 && other.Upper >= 0 {
		// Interval contains zero - return entire interval
		return Entire()
	}
	return i.Mul(Interval{Lower: 1 / other.Upper, Upper: 1 / other.Lower})
}

// Neg returns the negation of the interval.
func (i Interval) Neg() Interval {
	return Interval{Lower: -i.Upper, Upper: -i.Lower}
}

// Abs returns the absolute value interval.
func (i Interval) Abs() Interval {
	if i.Lower >= 0 {
		return i
	}
	if i.Upper <= 0 {
		return Interval{Lower: -i.Upper, Upper: -i.Lower}
	}
	return Interval{Lower: 0, Upper: maxFloat64(i.Upper, -i.Lower)}
}

// Mod returns the modulo interval (approximate).
func (i Interval) Mod(other Interval) Interval {
	if other.Lower <= 0 && other.Upper >= 0 {
		return Entire()
	}
	// Conservative approximation
	maxMod := maxFloat64(math.Abs(other.Lower), math.Abs(other.Upper))
	return Interval{Lower: 0, Upper: maxMod}
}

// Pow returns the interval raised to an integer power.
func (i Interval) Pow(n int) Interval {
	if n == 0 {
		return Point(1)
	}
	if n == 1 {
		return i
	}
	if n < 0 {
		return Point(1).Div(i.Pow(-n))
	}
	if n%2 == 0 {
		// Even power
		if i.Lower >= 0 {
			return Interval{
				Lower: math.Pow(i.Lower, float64(n)),
				Upper: math.Pow(i.Upper, float64(n)),
			}
		}
		if i.Upper <= 0 {
			return Interval{
				Lower: math.Pow(i.Upper, float64(n)),
				Upper: math.Pow(i.Lower, float64(n)),
			}
		}
		// Interval spans zero
		return Interval{
			Lower: 0,
			Upper: maxFloat64(math.Pow(i.Lower, float64(n)), math.Pow(i.Upper, float64(n))),
		}
	}
	// Odd power - monotonic
	return Interval{
		Lower: math.Pow(i.Lower, float64(n)),
		Upper: math.Pow(i.Upper, float64(n)),
	}
}

// Sqrt returns the square root interval.
func (i Interval) Sqrt() Interval {
	if i.Upper < 0 {
		return Empty()
	}
	lower := 0.0
	if i.Lower > 0 {
		lower = math.Sqrt(i.Lower)
	}
	return Interval{Lower: lower, Upper: math.Sqrt(i.Upper)}
}

// Exp returns the exponential interval.
func (i Interval) Exp() Interval {
	return Interval{
		Lower: math.Exp(i.Lower),
		Upper: math.Exp(i.Upper),
	}
}

// Log returns the natural logarithm interval.
func (i Interval) Log() Interval {
	if i.Upper <= 0 {
		return Empty()
	}
	lower := math.Inf(-1)
	if i.Lower > 0 {
		lower = math.Log(i.Lower)
	}
	return Interval{Lower: lower, Upper: math.Log(i.Upper)}
}

// Union returns the smallest interval containing both intervals.
func (i Interval) Union(other Interval) Interval {
	if i.IsEmpty() {
		return other
	}
	if other.IsEmpty() {
		return i
	}
	return Interval{
		Lower: minFloat64(i.Lower, other.Lower),
		Upper: maxFloat64(i.Upper, other.Upper),
	}
}

// Intersect returns the intersection of two intervals.
func (i Interval) Intersect(other Interval) Interval {
	if i.IsEmpty() || other.IsEmpty() {
		return Empty()
	}
	lower := maxFloat64(i.Lower, other.Lower)
	upper := minFloat64(i.Upper, other.Upper)
	if lower > upper {
		return Empty()
	}
	return Interval{Lower: lower, Upper: upper}
}

// BoolInterval represents a tristate Boolean interval.
type BoolInterval int

const (
	BoolIntervalUnknown BoolInterval = iota // Could be true or false
	BoolIntervalTrue                        // Definitely true
	BoolIntervalFalse                       // Definitely false
)

// String returns the string representation.
func (b BoolInterval) String() string {
	switch b {
	case BoolIntervalTrue:
		return "true"
	case BoolIntervalFalse:
		return "false"
	default:
		return "unknown"
	}
}

// IsResolved returns true if the boolean value is known.
func (b BoolInterval) IsResolved() bool {
	return b == BoolIntervalTrue || b == BoolIntervalFalse
}

// ToBool returns the boolean value if resolved.
func (b BoolInterval) ToBool() (bool, bool) {
	switch b {
	case BoolIntervalTrue:
		return true, true
	case BoolIntervalFalse:
		return false, true
	default:
		return false, false
	}
}

// And returns the Boolean AND of two intervals.
func (b BoolInterval) And(other BoolInterval) BoolInterval {
	if b == BoolIntervalFalse || other == BoolIntervalFalse {
		return BoolIntervalFalse
	}
	if b == BoolIntervalTrue && other == BoolIntervalTrue {
		return BoolIntervalTrue
	}
	return BoolIntervalUnknown
}

// Or returns the Boolean OR of two intervals.
func (b BoolInterval) Or(other BoolInterval) BoolInterval {
	if b == BoolIntervalTrue || other == BoolIntervalTrue {
		return BoolIntervalTrue
	}
	if b == BoolIntervalFalse && other == BoolIntervalFalse {
		return BoolIntervalFalse
	}
	return BoolIntervalUnknown
}

// Not returns the Boolean NOT of an interval.
func (b BoolInterval) Not() BoolInterval {
	switch b {
	case BoolIntervalTrue:
		return BoolIntervalFalse
	case BoolIntervalFalse:
		return BoolIntervalTrue
	default:
		return BoolIntervalUnknown
	}
}

// Compare returns the result of comparing two intervals.
type CompareResult int

const (
	CompareUnknown   CompareResult = iota // Comparison result is uncertain
	CompareLess                           // i < other is definitely true
	CompareEqual                          // i == other is definitely true
	CompareGreater                        // i > other is definitely true
	CompareNotEqual                       // i != other is definitely true
	CompareLessEq                         // i <= other is definitely true
	CompareGreaterEq                      // i >= other is definitely true
)

// Less returns whether i < other.
func (i Interval) Less(other Interval) BoolInterval {
	if i.Upper < other.Lower {
		return BoolIntervalTrue
	}
	if i.Lower >= other.Upper {
		return BoolIntervalFalse
	}
	return BoolIntervalUnknown
}

// LessEq returns whether i <= other.
func (i Interval) LessEq(other Interval) BoolInterval {
	if i.Upper <= other.Lower {
		return BoolIntervalTrue
	}
	if i.Lower > other.Upper {
		return BoolIntervalFalse
	}
	return BoolIntervalUnknown
}

// Greater returns whether i > other.
func (i Interval) Greater(other Interval) BoolInterval {
	return other.Less(i)
}

// GreaterEq returns whether i >= other.
func (i Interval) GreaterEq(other Interval) BoolInterval {
	return other.LessEq(i)
}

// Equal returns whether i == other (for point intervals).
func (i Interval) Equal(other Interval) BoolInterval {
	if i.IsPoint() && other.IsPoint() && i.Lower == other.Lower {
		return BoolIntervalTrue
	}
	if !i.Overlaps(other) {
		return BoolIntervalFalse
	}
	return BoolIntervalUnknown
}

// NotEqual returns whether i != other.
func (i Interval) NotEqual(other Interval) BoolInterval {
	return i.Equal(other).Not()
}

// Helper functions
func minFloat64(vals ...float64) float64 {
	min := vals[0]
	for _, v := range vals[1:] {
		if v < min || math.IsNaN(min) {
			min = v
		}
	}
	return min
}

func maxFloat64(vals ...float64) float64 {
	max := vals[0]
	for _, v := range vals[1:] {
		if v > max || math.IsNaN(max) {
			max = v
		}
	}
	return max
}

// IntervalFromDistribution extracts an interval from a distribution.
func IntervalFromDistribution(d Distribution) Interval {
	if d == nil {
		return Entire()
	}
	if d.IsPoint() {
		if v, ok := d.PointValue(); ok {
			if f, ok := v.AsFloat(); ok {
				return Point(f)
			}
		}
	}
	if lower, upper, ok := d.Support(); ok {
		return NewInterval(lower, upper)
	}
	return Entire()
}
