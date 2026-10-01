package placement

import (
	"fmt"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/api/resource"
)

// MemoryTable supplies per-device memory for drivers that publish none. The
// TPU DRA driver, for instance, reports no capacity at all; HBM is fixed per
// generation, so an operator table keyed on its `tpuGen` attribute stands in
// for the capacity NVIDIA's driver reports directly. Zero value: no table.
type MemoryTable struct {
	// Attribute is the unqualified device attribute the table is keyed on.
	Attribute string
	// Bytes maps each attribute value to that device's memory.
	Bytes map[string]int64
}

// ParseMemoryTable reads "attr:value=quantity,value=quantity,...", e.g.
// "tpuGen:v5e=16Gi,v6e=32Gi,v7x=192Gi". Empty input is the zero table.
func ParseMemoryTable(spec string) (MemoryTable, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return MemoryTable{}, nil
	}
	attr, entries, ok := strings.Cut(spec, ":")
	attr = strings.TrimSpace(attr)
	if !ok || attr == "" {
		return MemoryTable{}, fmt.Errorf("memory table %q: want attr:value=quantity[,...]", spec)
	}
	table := MemoryTable{Attribute: attr, Bytes: map[string]int64{}}
	for _, entry := range strings.Split(entries, ",") {
		value, quantity, ok := strings.Cut(strings.TrimSpace(entry), "=")
		if !ok || strings.TrimSpace(value) == "" {
			return MemoryTable{}, fmt.Errorf("memory table entry %q: want value=quantity", entry)
		}
		parsed, err := resource.ParseQuantity(strings.TrimSpace(quantity))
		if err != nil || parsed.Value() <= 0 {
			return MemoryTable{}, fmt.Errorf("memory table entry %q: bad quantity", entry)
		}
		table.Bytes[strings.TrimSpace(value)] = parsed.Value()
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

// ValuesWithin lists, sorted, the attribute values whose memory satisfies a
// tier: at least floor bytes, and in the same whole-GiB bucket as the
// ceiling, which is how Catalog grouped the devices the tier was priced on.
func (t MemoryTable) ValuesWithin(floor, ceiling int64) []string {
	var values []string
	for value, bytes := range t.Bytes {
		if bytes >= floor && CeilGiB(bytes) == CeilGiB(ceiling) {
			values = append(values, value)
		}
	}
	sort.Strings(values)
	return values
}
