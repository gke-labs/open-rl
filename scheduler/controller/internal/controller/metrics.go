package controller

import (
	"context"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"sigs.k8s.io/controller-runtime/pkg/client"

	openrlv1alpha1 "github.com/gke-labs/open-rl/scheduler/controller/api/v1alpha1"
)

// The UUID label is spelled as DCGM spells it, so GPU metrics join on it.
var gpuHeldDesc = prometheus.NewDesc(
	"openrl_gpu_held",
	"1 for each GPU a placed workload holds. Present from placement to release, so its presence over time is the hold interval.",
	[]string{"run_id", "role", "pod", "node", "UUID"}, nil,
)

// gpuHeldCollector reads workload status at scrape time, so a released or
// deleted workload's series disappears with it.
type gpuHeldCollector struct {
	reader    client.Reader
	namespace string
}

func (c gpuHeldCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- gpuHeldDesc
}

func (c gpuHeldCollector) Collect(ch chan<- prometheus.Metric) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var workers openrlv1alpha1.WorkloadList
	if err := c.reader.List(ctx, &workers, client.InNamespace(c.namespace)); err != nil {
		ch <- prometheus.NewInvalidMetric(gpuHeldDesc, err)
		return
	}
	for i := range workers.Items {
		worker := &workers.Items[i]
		status := worker.Status
		if status.NodeName == "" || status.ReleasedAt != nil {
			continue
		}
		for _, uuid := range status.GPUUUIDs {
			ch <- prometheus.MustNewConstMetric(gpuHeldDesc, prometheus.GaugeValue, 1,
				worker.Spec.ModelID, string(worker.Spec.Role), status.PodName, status.NodeName, uuid)
		}
	}
}
