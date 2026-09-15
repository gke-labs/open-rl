// Command manager runs the Open-RL placement controller.
//
// It watches Workload resources and reconciles each into a DRA
// ResourceClaim and a worker pod, letting Kubernetes pick the devices and the
// node. See scheduler/docs/design.md.
package main

import (
	"flag"
	"os"
	"strconv"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	openrlv1alpha1 "github.com/gke-labs/open-rl/scheduler/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/scheduler/controller/internal/controller"
	"github.com/gke-labs/open-rl/scheduler/controller/internal/placement"
)

var scheme = runtime.NewScheme()

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(openrlv1alpha1.AddToScheme(scheme))
	// +kubebuilder:scaffold:scheme
}

func main() {
	var (
		metricsAddr      string
		probeAddr        string
		leaderElection   bool
		namespace        string
		deviceClass      string
		deviceDriver     string
		retryInterval    time.Duration
		placementTimeout time.Duration
		strategy         string
		maxConcurrent    int
	)

	flag.StringVar(&metricsAddr, "metrics-bind-address", "0", "Address the metric endpoint binds to; 0 disables it.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "Address the probe endpoint binds to.")
	flag.BoolVar(&leaderElection, "leader-elect", true,
		"Hold a lease before placing. Two controllers placing at once would each decide against a fleet missing the other's bookings.")
	flag.StringVar(&namespace, "namespace", env("OPEN_RL_WORKER_NAMESPACE", "openrl-system"), "Namespace holding workers, claims and pods.")
	flag.StringVar(&deviceClass, "device-class", env("OPEN_RL_DEVICE_CLASS", "gpu.nvidia.com"), "DeviceClass generated claims request.")
	flag.StringVar(&deviceDriver, "device-driver", env("OPEN_RL_DEVICE_DRIVER", ""), "Driver publishing the ResourceSlices. Defaults to the device class.")
	flag.DurationVar(&retryInterval, "retry-interval", envDuration("OPEN_RL_RECONCILE_INTERVAL", 10*time.Second), "How often an unplaced worker is retried.")
	flag.DurationVar(&placementTimeout, "placement-timeout", envDuration("OPEN_RL_PLACEMENT_TIMEOUT", 15*time.Minute),
		"How long a worker may go unplaced before the request is declared unsatisfiable. 0 waits forever.")
	flag.StringVar(&strategy, "placement-strategy", env("OPEN_RL_PLACEMENT_STRATEGY", string(placement.StrategyBinPack)),
		"binpack shares eligible FFT claims first; spread requests a GPU first, then falls back to FFT sharing. LoRA never shares.")
	flag.IntVar(&maxConcurrent, "max-concurrent-reconciles", envInt("OPEN_RL_MAX_CONCURRENT_RECONCILES", 4),
		"How many workers place at once. Seat booking is CAS-arbitrated, so concurrency risks only transient over-cut claims, which the sharing fallback drains.")

	opts := zap.Options{Development: false}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))
	setupLog := ctrl.Log.WithName("setup")

	if deviceDriver == "" {
		deviceDriver = deviceClass
	}
	parsedStrategy, err := placement.ParseStrategy(strategy)
	if err != nil {
		setupLog.Error(err, "invalid placement strategy")
		os.Exit(1)
	}

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 scheme,
		Metrics:                metricsserver.Options{BindAddress: metricsAddr},
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         leaderElection,
		LeaderElectionID:       "placement.openrl.io",
		// Workers, claims, pods and ConfigMaps are all namespaced; nodes and
		// ResourceSlices are cluster-scoped. Nodes are cached only if the
		// operator opted them in: placement never reads any other node, and an
		// unfiltered informer would deliver every kubelet heartbeat in the
		// cluster to this controller's watch.
		Cache: cache.Options{
			DefaultNamespaces: map[string]cache.Config{namespace: {}},
			ByObject: map[client.Object]cache.ByObject{
				&corev1.Node{}: {Label: labels.SelectorFromSet(labels.Set{controller.NodeLabelEnabled: "true"})},
				// Pods from every namespace: the host-memory fit reserves what
				// system pods and exporters already request on each node.
				&corev1.Pod{}: {Namespaces: map[string]cache.Config{cache.AllNamespaces: {}}},
			},
		},
	})
	if err != nil {
		setupLog.Error(err, "cannot start manager")
		os.Exit(1)
	}

	reconciler := &controller.WorkloadReconciler{
		Client:                  mgr.GetClient(),
		Recorder:                mgr.GetEventRecorderFor("scheduler"),
		Namespace:               namespace,
		DeviceClass:             deviceClass,
		DeviceDriver:            deviceDriver,
		RetryInterval:           retryInterval,
		PlacementTimeout:        placementTimeout,
		PlacementStrategy:       parsedStrategy,
		MaxConcurrentReconciles: maxConcurrent,
	}
	if err := reconciler.SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "cannot set up the Workload controller")
		os.Exit(1)
	}
	// +kubebuilder:scaffold:builder

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "cannot add health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "cannot add ready check")
		os.Exit(1)
	}

	setupLog.Info("placing workers", "namespace", namespace, "deviceClass", deviceClass, "deviceDriver", deviceDriver, "strategy", strategy)
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "manager exited")
		os.Exit(1)
	}
}

func env(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

// envDuration reads a Go duration ("30s", "15m") from the environment.
func envDuration(key string, fallback time.Duration) time.Duration {
	if parsed, err := time.ParseDuration(os.Getenv(key)); err == nil {
		return parsed
	}
	return fallback
}

// envInt reads an integer from the environment.
func envInt(key string, fallback int) int {
	if parsed, err := strconv.Atoi(os.Getenv(key)); err == nil {
		return parsed
	}
	return fallback
}
