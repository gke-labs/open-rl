package controller

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/api/resource"

	"github.com/gke-labs/open-rl/scheduler/controller/internal/placement"
)

// DefaultTPUMemoryTable is HBM per chip by TPU generation, keyed on the TPU
// DRA driver's tpuGen attribute. That driver publishes no memory capacity.
const DefaultTPUMemoryTable = "tpuGen:v5e=16Gi,v6e=32Gi,v5p=95Gi,v7x=192Gi"

// celIdentifier is what a device attribute name must be to sit in CEL as
// device.attributes["driver"].<name>.
var celIdentifier = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// MemoryTable supplies per-device memory, by attribute value, for drivers
// that publish none. Zero value: no table.
type MemoryTable struct {
	// Attribute is the unqualified device attribute the table is keyed on.
	Attribute string
	// Bytes maps each attribute value to that device's memory.
	Bytes map[string]int64
}

// ParseMemoryTable reads "attr:value=quantity,value=quantity,...", e.g.
// "tpuGen:v5e=16Gi,v6e=32Gi". Empty input is the zero table.
func ParseMemoryTable(spec string) (MemoryTable, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return MemoryTable{}, nil
	}
	attr, entries, ok := strings.Cut(spec, ":")
	attr = strings.TrimSpace(attr)
	if !ok || !celIdentifier.MatchString(attr) {
		return MemoryTable{}, fmt.Errorf("memory table %q: want attr:value=quantity[,...] with attr an identifier", spec)
	}
	table := MemoryTable{Attribute: attr, Bytes: map[string]int64{}}
	for _, entry := range strings.Split(entries, ",") {
		value, quantity, ok := strings.Cut(entry, "=")
		value = strings.TrimSpace(value)
		if !ok || value == "" {
			return MemoryTable{}, fmt.Errorf("memory table entry %q: want value=quantity", entry)
		}
		if _, dup := table.Bytes[value]; dup {
			return MemoryTable{}, fmt.Errorf("memory table entry %q: %s listed twice", entry, value)
		}
		parsed, err := resource.ParseQuantity(strings.TrimSpace(quantity))
		if err != nil || parsed.Value() <= 0 {
			return MemoryTable{}, fmt.Errorf("memory table entry %q: want a positive quantity", entry)
		}
		table.Bytes[value] = parsed.Value()
	}
	return table, nil
}

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
