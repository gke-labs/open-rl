package sim

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/yaml"

	"github.com/gke-labs/open-rl/controller/internal/placement"
)

// Parse reads a scenario from YAML.
func Parse(data []byte) (Scenario, error) {
	var s Scenario
	if err := yaml.UnmarshalStrict(data, &s); err != nil {
		return Scenario{}, fmt.Errorf("parse scenario: %w", err)
	}
	if len(s.Nodes) == 0 {
		return Scenario{}, fmt.Errorf("scenario has no nodes")
	}
	if len(s.Work) == 0 {
		return Scenario{}, fmt.Errorf("scenario has no work")
	}
	return s, nil
}

// buildFleet turns the scenario's nodes into the Fleet the scheduler decides
// against. The claims start empty: the run creates them as workloads arrive,
// exactly as the controller would.
func buildFleet(nodes []NodeSpec) (*placement.Fleet, error) {
	fleet := placement.NewFleet()
	for _, n := range nodes {
		if n.Devices < 1 {
			return nil, fmt.Errorf("node %s: devices must be >= 1", n.Name)
		}
		deviceMemory, err := bytes(n.DeviceMemory)
		if err != nil {
			return nil, fmt.Errorf("node %s deviceMemory: %w", n.Name, err)
		}
		hostMemory, err := bytes(n.HostMemory)
		if err != nil {
			return nil, fmt.Errorf("node %s hostMemory: %w", n.Name, err)
		}
		roles := map[string]bool{}
		for _, role := range n.Roles {
			roles[role] = true
		}
		if len(roles) == 0 {
			return nil, fmt.Errorf("node %s accepts no roles, so nothing can land on it", n.Name)
		}
		fleet.Nodes[n.Name] = &placement.Node{
			Name:              n.Name,
			DeviceCount:       n.Devices,
			DeviceMemoryBytes: deviceMemory,
			HostMemoryBytes:   hostMemory,
			Roles:             roles,
			MaxResidents:      max(1, n.MaxResidents),
		}
	}
	return fleet, nil
}

// buildWorkers turns the scenario's workloads into the requests the scheduler
// sees, plus the two things only a simulation knows: when each arrives and how
// much accelerator time it needs.
func buildWorkers(specs []WorkloadSpec) ([]*worker, error) {
	seen := map[string]bool{}
	workers := make([]*worker, 0, len(specs))
	for _, w := range specs {
		if w.ID == "" {
			return nil, fmt.Errorf("every workload needs an id")
		}
		if seen[w.ID] {
			return nil, fmt.Errorf("workload id %q appears twice; ids are cohorts of one and must be unique", w.ID)
		}
		seen[w.ID] = true

		memory, err := bytes(w.Memory)
		if err != nil {
			return nil, fmt.Errorf("workload %s memory: %w", w.ID, err)
		}
		if memory <= 0 {
			return nil, fmt.Errorf("workload %s: memory must be > 0", w.ID)
		}
		at, err := duration(w.At, 0)
		if err != nil {
			return nil, fmt.Errorf("workload %s at: %w", w.ID, err)
		}
		work, err := duration(w.Work, 0)
		if err != nil {
			return nil, fmt.Errorf("workload %s work: %w", w.ID, err)
		}
		if w.Role != "trainer" && w.Role != "sampler" {
			return nil, fmt.Errorf("workload %s: role must be trainer or sampler, got %q", w.ID, w.Role)
		}

		workers = append(workers, &worker{
			spec: w, at: at, work: work,
			request: placement.Request{
				Role:     w.Role,
				Memory:   memory,
				Cohort:   w.Cohort,
				WorkerID: w.ID,
			},
		})
	}
	return workers, nil
}

// bytes parses a Kubernetes quantity like "24Gi". Empty is zero, which for a
// node's host memory means "not reported" and skips the parked-memory check.
func bytes(s string) (int64, error) {
	if s == "" {
		return 0, nil
	}
	q, err := resource.ParseQuantity(s)
	if err != nil {
		return 0, err
	}
	return q.Value(), nil
}

func duration(s string, fallback time.Duration) (time.Duration, error) {
	if s == "" {
		return fallback, nil
	}
	return time.ParseDuration(s)
}
