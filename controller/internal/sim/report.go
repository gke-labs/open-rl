package sim

import (
	"fmt"
	"io"
	"text/tabwriter"
	"time"
)

// Report writes a run as something you can read in a terminal: the timeline of
// what the scheduler did, then what it cost.
func Report(w io.Writer, res *Result, timeline bool) {
	if timeline {
		fmt.Fprintln(w, "TIMELINE")
		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		for _, e := range res.Events {
			fmt.Fprintf(tw, "  %s\t%s\t%s\n", short(e.At), e.Kind, e.Text)
		}
		tw.Flush()
		fmt.Fprintln(w)
	}

	fmt.Fprintln(w, "WORKLOADS")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "  ID\tCOHORT\tCLAIM\tNODE\tGPUS\tWAITED\tRAN\tNEEDED\tSWITCHES\tSTATE")
	for _, r := range res.Workers {
		state := "running"
		switch {
		case r.Done:
			state = "done"
		case r.Claim == "":
			state = "unplaced"
		}
		fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t%d\t%s\n",
			r.ID, or(r.Cohort, "-"), or(r.Claim, "-"), or(r.Node, "-"), r.Devices,
			short(r.Waited), short(r.Ran), short(r.Needed), r.Switches, state)
	}
	tw.Flush()

	fmt.Fprintln(w, "\nCLAIMS")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "  NAME\tNODE\tGPUS\tCOHORTS\tBUSY\tSWITCHING\tOVERHEAD")
	for _, c := range res.Claims {
		total := c.Busy + c.Switching
		overhead := "0%"
		if total > 0 {
			overhead = fmt.Sprintf("%.0f%%", 100*float64(c.Switching)/float64(total))
		}
		fmt.Fprintf(tw, "  %s\t%s\t%d\t%d\t%s\t%s\t%s\n",
			c.Name, or(c.Node, "-"), c.Devices, c.Cohorts, short(c.Busy), short(c.Switching), overhead)
	}
	tw.Flush()

	if len(res.Unplaced) > 0 {
		fmt.Fprintln(w, "\nUNPLACED")
		for _, u := range res.Unplaced {
			fmt.Fprintf(w, "  %s: %s\n", u.ID, u.Reason)
		}
	}

	fmt.Fprintf(w, "\nran for %s\n", short(res.Elapsed))
}

func short(d time.Duration) string {
	if d == 0 {
		return "0s"
	}
	return d.Round(time.Second).String()
}

func or(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}
