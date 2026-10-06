package controller

import (
	"sort"

	"github.com/gke-labs/open-rl/scheduler/controller/internal/placement"
)

// MemoryTable supplies per-device memory, by attribute value, for drivers
// that publish none. Zero value: no table.
type MemoryTable struct {
	// Attribute is the unqualified device attribute the table is keyed on.
	Attribute string
	// Bytes maps each attribute value to that device's memory.
	Bytes map[string]int64
}

// DefaultTPUMemoryTable is HBM per chip by TPU generation, keyed on the TPU
// DRA driver's tpuGen attribute. That driver publishes no memory capacity.
var DefaultTPUMemoryTable = MemoryTable{Attribute: "tpuGen", Bytes: map[string]int64{
	"v5e": 16 * placement.GiB,
	"v6e": 32 * placement.GiB,
	"v5p": 95 * placement.GiB,
	"v7x": 192 * placement.GiB,
}}

// Enabled reports whether a table was configured.
func (t MemoryTable) Enabled() bool { return t.Attribute != "" && len(t.Bytes) > 0 }

// Lookup is the memory for one attribute value.
func (t MemoryTable) Lookup(value string) (int64, bool) {
	bytes, ok := t.Bytes[value]
	return bytes, ok
}

// ValuesWithin lists, sorted, the values whose memory satisfies a tier: at
// least floor bytes, and in the ceiling's whole-GiB bucket, which is how
// Catalog grouped the devices the tier was priced on.
func (t MemoryTable) ValuesWithin(floor, ceiling int64) []string {
	var values []string
	for value, bytes := range t.Bytes {
		if bytes >= floor && placement.CeilGiB(bytes) == placement.CeilGiB(ceiling) {
			values = append(values, value)
		}
	}
	sort.Strings(values)
	return values
}
