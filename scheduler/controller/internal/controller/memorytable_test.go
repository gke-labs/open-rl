package controller

import (
	"reflect"
	"testing"

	"github.com/gke-labs/open-rl/scheduler/controller/internal/placement"
)

const gib = placement.GiB

func TestMemoryTableLookup(t *testing.T) {
	if bytes, ok := DefaultTPUMemoryTable.Lookup("v6e"); !ok || bytes != 32*gib || !DefaultTPUMemoryTable.Enabled() {
		t.Errorf("Lookup(v6e) = %d, %v; want 32Gi from an enabled table", bytes, ok)
	}
	if _, ok := DefaultTPUMemoryTable.Lookup("v4"); ok {
		t.Error("Lookup(v4) found a value the table does not have")
	}
	if (MemoryTable{}).Enabled() {
		t.Error("the zero table is enabled")
	}
}

// ValuesWithin admits the values a tier was priced on: in the ceiling's
// whole-gib bucket, as Catalog groups sizes, and at least the floor.
func TestMemoryTableValuesWithin(t *testing.T) {
	table := MemoryTable{Attribute: "gen", Bytes: map[string]int64{
		"a": 16 * gib,
		"b": 32 * gib,
		"c": 32 * gib,
		"d": 31*gib + gib/2, // same 32Gi bucket, a little smaller
		"e": 95 * gib,
	}}
	for _, tc := range []struct {
		name           string
		floor, ceiling int64
		want           []string
	}{
		{"bucket, sorted", 20 * gib, 32 * gib, []string{"b", "c", "d"}},
		{"floor drops the smaller device in the bucket", 31*gib + 3*gib/4, 32 * gib, []string{"b", "c"}},
		{"other buckets excluded", 0, 95 * gib, []string{"e"}},
		{"floor above every device", 33 * gib, 32 * gib, nil},
		{"no device in the bucket", 1, 48 * gib, nil},
	} {
		if got := table.ValuesWithin(tc.floor, tc.ceiling); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: ValuesWithin = %v, want %v", tc.name, got, tc.want)
		}
	}
}
