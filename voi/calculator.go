// Package voi implements Value of Information calculations for input selection.
package voi

import (
	"math"
	"math/rand"
	"sort"

	"github.com/karpathy/dag-evaluator/dag"
	"github.com/karpathy/dag-evaluator/distribution"
	"github.com/karpathy/dag-evaluator/propagation"
)

// UtilityFunction defines how utility is computed from a distribution.
type UtilityFunction int

const (
	// UtilityEntropy uses negative entropy (minimize uncertainty).
	UtilityEntropy UtilityFunction = iota
	// UtilityVariance uses negative variance (minimize spread).
	UtilityVariance
	// UtilityConfidence uses classification confidence for Boolean outcomes.
	UtilityConfidence
	// UtilityIntervalWidth uses negative interval width.
	UtilityIntervalWidth
)

// Config holds configuration for VoI calculations.
type Config struct {
	// NumVoISamples is the number of samples for computing expected VoI.
	NumVoISamples int
	// NumPropSamples is the number of samples for propagation after observation.
	NumPropSamples int
	// UtilityFunc defines the utility measure to use.
	UtilityFunc UtilityFunction
	// CostWeight (lambda) is the weight applied to acquisition cost.
	// If 0, uses ratio-based scoring (VoI/cost).
	CostWeight float64
	// RNG is the random number generator.
	RNG *rand.Rand
}

// DefaultConfig returns a default configuration.
func DefaultConfig() Config {
	return Config{
		NumVoISamples:  100,
		NumPropSamples: 500,
		UtilityFunc:    UtilityConfidence,
		CostWeight:     1.0,
		RNG:            rand.New(rand.NewSource(42)),
	}
}

// Calculator computes Value of Information for inputs.
type Calculator struct {
	config Config
	dag    *dag.DAG
}

// NewCalculator creates a new VoI calculator.
func NewCalculator(d *dag.DAG, cfg Config) *Calculator {
	return &Calculator{
		config: cfg,
		dag:    d,
	}
}

// InputScore represents the VoI score for an input.
type InputScore struct {
	InputID       string
	VoI           float64 // Value of Information
	Cost          float64 // Acquisition cost
	NetScore      float64 // VoI - lambda*Cost or VoI/Cost
	CurrentUtil   float64 // Current utility before acquisition
	ExpectedUtil  float64 // Expected utility after acquisition
	Sensitivity   float64 // How much root depends on this input
}

// ComputeAllScores computes VoI scores for all unresolved inputs.
func (c *Calculator) ComputeAllScores() []InputScore {
	var scores []InputScore

	// Get current utility
	currentUtil := c.computeCurrentUtility()

	// Get unresolved inputs
	unresolvedInputs := c.dag.GetUnresolvedInputNodes()

	// Compute sensitivity to pre-filter inputs
	intervalProp := propagation.NewIntervalPropagator(c.dag)
	intervalProp.PropagateIntervals() // Ensure intervals are populated
	sensitivity := intervalProp.ComputeSensitivity()

	for _, input := range unresolvedInputs {
		// Skip inputs that cannot affect the root
		if !intervalProp.CanAffectRoot(input.ID) {
			continue
		}

		score := c.computeInputScore(input, currentUtil, sensitivity[input.ID])
		scores = append(scores, score)
	}

	// Sort by net score (highest first)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].NetScore > scores[j].NetScore
	})

	return scores
}

// SelectNextInput selects the best input to acquire.
func (c *Calculator) SelectNextInput() (*InputScore, bool) {
	scores := c.ComputeAllScores()
	if len(scores) == 0 {
		return nil, false
	}

	// Return the input with highest net score if positive
	best := &scores[0]
	if best.NetScore > 0 {
		return best, true
	}

	// If all scores are non-positive, still return the best one
	// (caller can decide whether to acquire)
	return best, true
}

func (c *Calculator) computeInputScore(node *dag.Node, currentUtil, sensitivity float64) InputScore {
	score := InputScore{
		InputID:     node.ID,
		Cost:        node.GetCost(),
		CurrentUtil: currentUtil,
		Sensitivity: sensitivity,
	}

	// Quick pruning: if sensitivity is very low, skip expensive VoI calculation
	if sensitivity < 0.001 {
		score.VoI = 0
		score.ExpectedUtil = currentUtil
		score.NetScore = -score.Cost * c.config.CostWeight
		return score
	}

	// Compute expected utility after acquiring this input
	expectedUtil := c.computeExpectedUtilityAfterAcquisition(node)
	score.ExpectedUtil = expectedUtil
	score.VoI = expectedUtil - currentUtil

	// Compute net score
	if c.config.CostWeight == 0 {
		// Ratio-based scoring
		if score.Cost > 0 {
			score.NetScore = score.VoI / score.Cost
		} else {
			score.NetScore = score.VoI * 1e6 // Very high score for free inputs
		}
	} else {
		// Difference-based scoring
		score.NetScore = score.VoI - c.config.CostWeight*score.Cost
	}

	return score
}

func (c *Calculator) computeCurrentUtility() float64 {
	root := c.dag.GetRoot()
	if root == nil || root.Distribution == nil {
		return 0
	}
	return c.utilityFromDistribution(root.Distribution)
}

func (c *Calculator) computeExpectedUtilityAfterAcquisition(input *dag.Node) float64 {
	if input.Distribution == nil || input.Input == nil {
		return c.computeCurrentUtility()
	}

	prior := input.Input.Prior
	if prior == nil {
		prior = input.Distribution
	}

	totalUtil := 0.0
	validSamples := 0

	for i := 0; i < c.config.NumVoISamples; i++ {
		// Sample a hypothetical observation from the prior
		observation := prior.Sample(c.config.RNG)

		// Compute utility under this observation
		util, err := c.computeUtilityWithObservation(input.ID, observation)
		if err != nil {
			continue
		}

		totalUtil += util
		validSamples++
	}

	if validSamples == 0 {
		return c.computeCurrentUtility()
	}

	return totalUtil / float64(validSamples)
}

func (c *Calculator) computeUtilityWithObservation(inputID string, observation distribution.Value) (float64, error) {
	// Clone the DAG
	clonedDAG := c.dag.Clone()

	// Set the observation
	node, ok := clonedDAG.GetNode(inputID)
	if !ok {
		return 0, nil
	}
	node.SetResolved(observation)

	// Propagate through the cloned DAG
	propConfig := propagation.Config{
		NumSamples: c.config.NumPropSamples,
		RNG:        c.config.RNG,
	}
	prop := propagation.NewPropagator(clonedDAG, propConfig)

	if err := prop.PropagateAll(); err != nil {
		return 0, err
	}

	// Compute utility from root distribution
	root := clonedDAG.GetRoot()
	if root == nil || root.Distribution == nil {
		return 0, nil
	}

	return c.utilityFromDistribution(root.Distribution), nil
}

func (c *Calculator) utilityFromDistribution(d distribution.Distribution) float64 {
	switch c.config.UtilityFunc {
	case UtilityEntropy:
		return -d.Entropy()

	case UtilityVariance:
		if v, ok := d.Variance(); ok {
			return -v
		}
		return -d.Entropy()

	case UtilityConfidence:
		return c.computeConfidenceUtility(d)

	case UtilityIntervalWidth:
		if lower, upper, ok := d.Support(); ok {
			width := upper - lower
			if math.IsInf(width, 0) || math.IsNaN(width) {
				return -1e12
			}
			return -width
		}
		return -d.Entropy()

	default:
		return -d.Entropy()
	}
}

func (c *Calculator) computeConfidenceUtility(d distribution.Distribution) float64 {
	// For Boolean distributions
	if bd, ok := d.(*distribution.BoolDistribution); ok {
		return bd.Confidence()
	}

	// For empirical distributions, check if samples are Boolean
	if ed, ok := d.(*distribution.EmpiricalDistribution); ok {
		if len(ed.Samples) > 0 && ed.Samples[0].Type == distribution.ValueTypeBool {
			// Count true/false
			trueCount := 0
			for _, s := range ed.Samples {
				if s.Bool {
					trueCount++
				}
			}
			pTrue := float64(trueCount) / float64(len(ed.Samples))
			if pTrue > 0.5 {
				return pTrue
			}
			return 1 - pTrue
		}

		// For categorical/enum distributions
		if len(ed.Samples) > 0 && (ed.Samples[0].Type == distribution.ValueTypeEnum || ed.Samples[0].Type == distribution.ValueTypeString) {
			// Find most common value
			counts := make(map[string]int)
			for _, s := range ed.Samples {
				counts[s.String]++
			}
			maxCount := 0
			for _, count := range counts {
				if count > maxCount {
					maxCount = count
				}
			}
			return float64(maxCount) / float64(len(ed.Samples))
		}
	}

	// For point distributions
	if d.IsPoint() {
		return 1.0
	}

	// For numeric distributions, use variance-based confidence
	if v, ok := d.Variance(); ok {
		// Lower variance = higher confidence
		// Use exponential decay
		return math.Exp(-v)
	}

	return 0.5
}

// ResolutionObjective defines the goal for evaluation.
type ResolutionObjective struct {
	// Type specifies the resolution type.
	Type ResolutionType
	// Threshold for comparison objectives (e.g., Y > T).
	Threshold float64
	// Alpha is the confidence level required.
	Alpha float64
	// IntervalWidth is the maximum acceptable interval width.
	IntervalWidth float64
}

// ResolutionType specifies what kind of resolution is needed.
type ResolutionType int

const (
	// ResolveToValue requires determining the exact value.
	ResolveToValue ResolutionType = iota
	// ResolveToInterval requires narrowing to an interval.
	ResolveToInterval
	// ResolveGreaterThan requires P(Y > T) > alpha.
	ResolveGreaterThan
	// ResolveLessThan requires P(Y < T) > alpha.
	ResolveLessThan
	// ResolveBoolean requires determining true/false with confidence.
	ResolveBoolean
)

// IsResolved checks if the current state satisfies the resolution objective.
func (c *Calculator) IsResolved(obj ResolutionObjective) bool {
	root := c.dag.GetRoot()
	if root == nil {
		return false
	}

	// If root is a point distribution, it's always resolved
	if root.IsResolved || (root.Distribution != nil && root.Distribution.IsPoint()) {
		return true
	}

	switch obj.Type {
	case ResolveToValue:
		return root.IsResolved

	case ResolveToInterval:
		lower, upper, ok := root.Distribution.Support()
		if !ok {
			return false
		}
		return (upper - lower) <= obj.IntervalWidth

	case ResolveGreaterThan:
		return c.checkProbabilityThreshold(root.Distribution, obj.Threshold, true, obj.Alpha)

	case ResolveLessThan:
		return c.checkProbabilityThreshold(root.Distribution, obj.Threshold, false, obj.Alpha)

	case ResolveBoolean:
		if bd, ok := root.Distribution.(*distribution.BoolDistribution); ok {
			return bd.Confidence() >= obj.Alpha
		}
		// For empirical distributions
		if ed, ok := root.Distribution.(*distribution.EmpiricalDistribution); ok {
			if len(ed.Samples) > 0 && ed.Samples[0].Type == distribution.ValueTypeBool {
				trueCount := 0
				for _, s := range ed.Samples {
					if s.Bool {
						trueCount++
					}
				}
				pTrue := float64(trueCount) / float64(len(ed.Samples))
				confidence := pTrue
				if pTrue < 0.5 {
					confidence = 1 - pTrue
				}
				return confidence >= obj.Alpha
			}
		}
		return false

	default:
		return root.IsResolved
	}
}

func (c *Calculator) checkProbabilityThreshold(d distribution.Distribution, threshold float64, greaterThan bool, alpha float64) bool {
	ed, ok := d.(*distribution.EmpiricalDistribution)
	if !ok {
		return false
	}

	count := 0
	for _, s := range ed.Samples {
		f, ok := s.AsFloat()
		if !ok {
			continue
		}
		if greaterThan && f > threshold {
			count++
		} else if !greaterThan && f < threshold {
			count++
		}
	}

	prob := float64(count) / float64(len(ed.Samples))
	return prob >= alpha
}

// GetCurrentConfidence returns the current confidence level at the root.
func (c *Calculator) GetCurrentConfidence() float64 {
	root := c.dag.GetRoot()
	if root == nil || root.Distribution == nil {
		return 0
	}
	return c.computeConfidenceUtility(root.Distribution)
}

// GetExpectedTotalCost estimates the total cost to resolve the DAG.
// Uses a greedy heuristic.
func (c *Calculator) GetExpectedTotalCost(obj ResolutionObjective) float64 {
	if c.IsResolved(obj) {
		return 0
	}

	totalCost := 0.0
	clonedDAG := c.dag.Clone()
	clonedCalc := NewCalculator(clonedDAG, c.config)

	maxIterations := len(clonedDAG.GetUnresolvedInputNodes()) + 1

	for i := 0; i < maxIterations; i++ {
		if clonedCalc.IsResolved(obj) {
			break
		}

		score, found := clonedCalc.SelectNextInput()
		if !found {
			break
		}

		totalCost += score.Cost

		// Simulate acquiring the input (use mean value as estimate)
		node, ok := clonedDAG.GetNode(score.InputID)
		if !ok {
			continue
		}

		// Use mean of prior as hypothetical observation
		var observation distribution.Value
		if mean, ok := node.Distribution.Mean(); ok {
			observation = distribution.NewFloatValue(mean)
		} else {
			// For non-numeric, sample once
			observation = node.Distribution.Sample(c.config.RNG)
		}
		node.SetResolved(observation)

		// Re-propagate
		propConfig := propagation.Config{
			NumSamples: c.config.NumPropSamples,
			RNG:        c.config.RNG,
		}
		prop := propagation.NewPropagator(clonedDAG, propConfig)
		prop.PropagateAll()
	}

	return totalCost
}
