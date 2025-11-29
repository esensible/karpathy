// Command dag-eval provides a CLI for evaluating DAG expression schemas.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/karpathy/dag-evaluator/dag"
	"github.com/karpathy/dag-evaluator/distribution"
	"github.com/karpathy/dag-evaluator/evaluator"
	"github.com/karpathy/dag-evaluator/schema"
	"github.com/karpathy/dag-evaluator/voi"
)

func main() {
	// Parse flags
	schemaPath := flag.String("schema", "", "Path to schema JSON file")
	interactive := flag.Bool("interactive", false, "Run in interactive mode")
	autoMode := flag.Bool("auto", false, "Run automatically with random inputs")
	budget := flag.Float64("budget", 0, "Maximum budget for acquisitions (0 = unlimited)")
	confidence := flag.Float64("confidence", 0.95, "Target confidence level")
	samples := flag.Int("samples", 1000, "Number of Monte Carlo samples")
	seed := flag.Int64("seed", 0, "Random seed (0 = use current time)")
	verbose := flag.Bool("verbose", false, "Verbose output")
	showScores := flag.Bool("scores", false, "Show VoI scores for all inputs")
	flag.Parse()

	if *schemaPath == "" {
		fmt.Println("Usage: dag-eval -schema <path> [-interactive] [-auto] [-budget <amount>]")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Load schema
	s, err := schema.Load(*schemaPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading schema: %v\n", err)
		os.Exit(1)
	}

	if *verbose {
		fmt.Printf("Loaded schema: %s\n", s.Name)
		fmt.Printf("  Description: %s\n", s.Description)
		fmt.Printf("  Root node: %s\n", s.RootNodeID)
		fmt.Printf("  Inputs: %d\n", len(s.Inputs))
		fmt.Printf("  Expressions: %d\n", len(s.Expressions))
		fmt.Println()
	}

	// Set up RNG
	var rng *rand.Rand
	if *seed == 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	} else {
		rng = rand.New(rand.NewSource(*seed))
	}

	// Create evaluator config
	cfg := evaluator.Config{
		NumSamples:    *samples,
		NumVoISamples: 100,
		UtilityFunc:   voi.UtilityConfidence,
		CostWeight:    1.0,
		MaxIterations: 100,
		RNG:           rng,
		Objective: voi.ResolutionObjective{
			Type:  voi.ResolveBoolean,
			Alpha: *confidence,
		},
	}

	// Create evaluator
	eval, err := evaluator.New(s, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating evaluator: %v\n", err)
		os.Exit(1)
	}

	// Initialize (propagate with priors)
	if err := eval.Initialize(); err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing evaluator: %v\n", err)
		os.Exit(1)
	}

	if *verbose {
		fmt.Printf("Initial confidence: %.2f%%\n", eval.GetCurrentConfidence()*100)
		fmt.Println()
	}

	// Show scores if requested
	if *showScores {
		showAllScores(eval)
	}

	// Run evaluation
	var result *evaluator.Result
	if *interactive {
		result, err = runInteractive(eval, s, *budget, *verbose)
	} else if *autoMode {
		result, err = eval.RunAutomatic(evaluator.RandomProvider(rng), *budget)
	} else {
		// Just show initial state
		result = eval.GetResult()
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error during evaluation: %v\n", err)
		os.Exit(1)
	}

	// Print result
	printResult(result, *verbose)
}

func showAllScores(eval *evaluator.Evaluator) {
	scores := eval.GetAllScores()
	fmt.Println("Value of Information Scores:")
	fmt.Println("============================")
	for i, score := range scores {
		fmt.Printf("%d. %s\n", i+1, score.InputID)
		fmt.Printf("   VoI: %.4f, Cost: %.2f, Net Score: %.4f\n", score.VoI, score.Cost, score.NetScore)
		fmt.Printf("   Sensitivity: %.4f\n", score.Sensitivity)
		fmt.Println()
	}
}

func runInteractive(eval *evaluator.Evaluator, s *schema.Schema, budget float64, verbose bool) (*evaluator.Result, error) {
	reader := bufio.NewReader(os.Stdin)

	for {
		// Check if resolved
		if eval.IsResolved() {
			fmt.Println("\nResolution objective achieved!")
			break
		}

		// Check budget
		if budget > 0 && eval.GetTotalCost() >= budget {
			fmt.Printf("\nBudget exhausted (%.2f / %.2f)\n", eval.GetTotalCost(), budget)
			break
		}

		// Get recommended next input
		score, found := eval.GetNextInput()
		if !found {
			fmt.Println("\nNo more inputs to acquire")
			break
		}

		// Display current state
		fmt.Printf("\nCurrent confidence: %.2f%%\n", eval.GetCurrentConfidence()*100)
		fmt.Printf("Total cost so far: %.2f\n", eval.GetTotalCost())
		fmt.Println()

		// Display recommendation
		node, _ := eval.GetInput(score.InputID)
		fmt.Printf("Recommended input: %s\n", score.InputID)
		if node.Input != nil && node.Input.Label != "" {
			fmt.Printf("  Label: %s\n", node.Input.Label)
		}
		if node.Description != "" {
			fmt.Printf("  Description: %s\n", node.Description)
		}
		fmt.Printf("  Cost: %.2f\n", score.Cost)
		fmt.Printf("  Expected VoI: %.4f\n", score.VoI)
		fmt.Printf("  Net Score: %.4f\n", score.NetScore)
		fmt.Println()

		// Ask for input
		question := score.InputID
		if node.Input != nil && node.Input.Question != "" {
			question = node.Input.Question
		}
		fmt.Printf("%s: ", question)

		input, err := reader.ReadString('\n')
		if err != nil {
			return nil, err
		}
		input = strings.TrimSpace(input)

		// Handle special commands
		if input == "skip" || input == "s" {
			fmt.Println("Skipping this input...")
			continue
		}
		if input == "quit" || input == "q" {
			fmt.Println("Quitting...")
			break
		}
		if input == "scores" {
			showAllScores(eval)
			continue
		}

		// Parse the input value
		value, err := parseInputValue(input, node, s)
		if err != nil {
			fmt.Printf("Error parsing input: %v\n", err)
			continue
		}

		// Acquire the input
		if err := eval.AcquireInput(score.InputID, value); err != nil {
			fmt.Printf("Error acquiring input: %v\n", err)
			continue
		}

		if verbose {
			fmt.Printf("Acquired %s = %s\n", score.InputID, value.AsString())
		}
	}

	return eval.GetResult(), nil
}

func parseInputValue(input string, node *dag.Node, s *schema.Schema) (distribution.Value, error) {
	// Check for enum type
	if node.Input != nil && node.Input.EnumType != "" {
		// Validate enum value
		values, ok := s.GetEnumValues(node.Input.EnumType)
		if ok {
			for _, v := range values {
				if strings.EqualFold(input, v) {
					return distribution.NewEnumValue(node.Input.EnumType, v), nil
				}
			}
			return distribution.Value{}, fmt.Errorf("invalid enum value, expected one of: %v", values)
		}
		return distribution.NewEnumValue(node.Input.EnumType, input), nil
	}

	// Try to parse based on data type
	switch node.DataType {
	case schema.DataTypeBool:
		b, err := strconv.ParseBool(input)
		if err != nil {
			return distribution.Value{}, fmt.Errorf("expected boolean (true/false)")
		}
		return distribution.NewBoolValue(b), nil

	case schema.DataTypeInt:
		i, err := strconv.ParseInt(input, 10, 64)
		if err != nil {
			return distribution.Value{}, fmt.Errorf("expected integer")
		}
		return distribution.NewIntValue(i), nil

	case schema.DataTypeFloat:
		f, err := strconv.ParseFloat(input, 64)
		if err != nil {
			return distribution.Value{}, fmt.Errorf("expected number")
		}
		return distribution.NewFloatValue(f), nil

	case schema.DataTypeString:
		return distribution.NewStringValue(input), nil

	default:
		// Try to infer type
		if b, err := strconv.ParseBool(input); err == nil {
			return distribution.NewBoolValue(b), nil
		}
		if f, err := strconv.ParseFloat(input, 64); err == nil {
			return distribution.NewFloatValue(f), nil
		}
		return distribution.NewStringValue(input), nil
	}
}

func printResult(result *evaluator.Result, verbose bool) {
	fmt.Println("\n========== EVALUATION RESULT ==========")
	fmt.Printf("Resolved: %v\n", result.IsResolved)
	fmt.Printf("Confidence: %.2f%%\n", result.Confidence*100)
	fmt.Printf("Total Cost: %.2f\n", result.TotalCost)
	fmt.Printf("Steps: %d\n", result.Steps)

	if result.Value.Type != 0 {
		fmt.Printf("Value: %s\n", result.Value.AsString())
	}

	if verbose && len(result.History) > 0 {
		fmt.Println("\nEvaluation History:")
		for _, step := range result.History {
			fmt.Printf("  %d. %s = %s (cost: %.2f, confidence: %.2f%%)\n",
				step.StepNumber,
				step.InputAcquired,
				step.InputValue.AsString(),
				step.Cost,
				step.Confidence*100,
			)
		}
	}

	// Output JSON for programmatic consumption
	if !verbose {
		jsonResult := map[string]interface{}{
			"resolved":   result.IsResolved,
			"confidence": result.Confidence,
			"total_cost": result.TotalCost,
			"steps":      result.Steps,
		}
		if result.Value.Type != 0 {
			jsonResult["value"] = result.Value.ToInterface()
		}
		jsonBytes, _ := json.Marshal(jsonResult)
		fmt.Printf("\nJSON: %s\n", string(jsonBytes))
	}
}

