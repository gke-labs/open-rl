package v1alpha1

import (
	"os"
	"reflect"
	"testing"

	"sigs.k8s.io/yaml"
)

// The checked-in CRD (make manifests) defaults spec.accelerator.type to GPU,
// so Workloads written before the field existed read as GPU workloads.
func TestCRDDefaultsAcceleratorTypeToGPU(t *testing.T) {
	raw, err := os.ReadFile("../../../deploy/base/00-workload-crd.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var crd map[string]any
	if err := yaml.Unmarshal(raw, &crd); err != nil {
		t.Fatal(err)
	}
	node := any(crd)
	for _, key := range []any{"spec", "versions", 0, "schema", "openAPIV3Schema", "properties", "spec",
		"properties", "accelerator", "properties", "type"} {
		switch k := key.(type) {
		case string:
			node = node.(map[string]any)[k]
		case int:
			node = node.([]any)[k]
		}
		if node == nil {
			t.Fatalf("CRD has no %v on the path to spec.accelerator.type", key)
		}
	}
	field := node.(map[string]any)
	if field["default"] != string(AcceleratorTypeGPU) {
		t.Errorf("default = %v, want GPU", field["default"])
	}
	if want := []any{"GPU", "TPU"}; !reflect.DeepEqual(field["enum"], want) {
		t.Errorf("enum = %v, want %v", field["enum"], want)
	}
}
