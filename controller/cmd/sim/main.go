// Command sim runs the scheduler against a made-up cluster.
//
//	go run ./cmd/sim scenarios/one-l4.yaml
//
// No Kubernetes, no GPUs, no images. It calls the same internal/placement
// functions the controller calls, so the placements it reports are the ones the
// real thing would make; what it adds is a clock, so you can also see what the
// turn-taking costs.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/gke-labs/open-rl/controller/internal/sim"
)

func main() {
	timeline := flag.Bool("timeline", true, "print every scheduling event in order")
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: sim [-timeline=false] SCENARIO.yaml")
		os.Exit(2)
	}

	data, err := os.ReadFile(flag.Arg(0))
	if err != nil {
		fail(err)
	}
	scenario, err := sim.Parse(data)
	if err != nil {
		fail(err)
	}
	result, err := sim.Run(scenario)
	if err != nil {
		fail(err)
	}
	sim.Report(os.Stdout, result, *timeline)
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "sim:", err)
	os.Exit(1)
}
