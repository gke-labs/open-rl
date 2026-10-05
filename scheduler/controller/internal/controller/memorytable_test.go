package controller

import (
	"reflect"
	"strings"
	"testing"

	"github.com/gke-labs/open-rl/scheduler/controller/internal/placement"
)

const gib = placement.GiB

func TestParseMemoryTable(t *testing.T) {
	table, err := ParseMemoryTable(" tpuGen : v5e=16Gi, v6e = 32Gi ,v7x=192Gi ")
	if err != nil {
		t.Fatal(err)
	}
	want := MemoryTable{Attribute: "tpuGen", Bytes: map[string]int64{"v5e": 16 * gib, "v6e": 32 * gib, "v7x": 192 * gib}}
	if !reflect.DeepEqual(table, want) || !table.Enabled() {
		t.Errorf("table = %+v, want %+v and enabled", table, want)
	}
	if bytes, ok := table.Lookup("v6e"); !ok || bytes != 32*gib {
		t.Errorf("Lookup(v6e) = %d, %v; want 32Gi", bytes, ok)
	}
	if _, ok := table.Lookup("v4"); ok {
		t.Error("Lookup(v4) found a value the table does not have")
	}

	empty, err := ParseMemoryTable("  ")
	if err != nil || empty.Enabled() {
		t.Errorf("empty spec = %+v, %v; want the disabled zero table", empty, err)
	}
	if _, ok := empty.Lookup("v6e"); ok {
		t.Error("the zero table looked something up")
	}
}

func TestParseMemoryTableRejectsMalformedSpecs(t *testing.T) {
	for _, spec := range []string{
		"v6e=32Gi",               // no attribute
		":v6e=32Gi",              // empty attribute
		"tpu.gen:v6e=32Gi",       // not an identifier; it lands in CEL
		"tpuGen:",                // no entries
		"tpuGen:v6e",             // no quantity
		"tpuGen:=32Gi",           // no value
		"tpuGen:v6e=lots",        // bad quantity
		"tpuGen:v6e=0",           // non-positive
		"tpuGen:v6e=32Gi,v6e=16", // duplicate value
	} {
		if table, err := ParseMemoryTable(spec); err == nil {
			t.Errorf("ParseMemoryTable(%q) = %+v, want an error", spec, table)
		}
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

func TestParseMemoryTableErrorNamesTheEntry(t *testing.T) {
	_, err := ParseMemoryTable("tpuGen:v6e=32Gi,v7x=huge")
	if err == nil || !strings.Contains(err.Error(), "v7x=huge") {
		t.Errorf("err = %v, want it to name the bad entry", err)
	}
}
