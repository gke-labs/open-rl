// Command manager runs the Open-RL placement controller.
//
// It watches OpenRLWorker resources and reconciles each into a DRA
// ResourceClaim and a worker pod, letting Kubernetes pick the devices and the
// node. See docs/designs/012-dynamic-placement.md.
package main

import (
	"flag"
	"os"
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

	openrlv1alpha1 "github.com/gke-labs/open-rl/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/controller/internal/controller"
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
		trainerTemplate  string
		samplerTemplate  string
		retryInterval    time.Duration
		placementTimeout time.Duration
		reclaimInterval  time.Duration
	)

	flag.StringVar(&metricsAddr, "metrics-bind-address", "0", "Address the metric endpoint binds to; 0 disables it.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "Address the probe endpoint binds to.")
	flag.BoolVar(&leaderElection, "leader-elect", true,
		"Hold a lease before placing. Two controllers placing at once would each decide against a fleet missing the other's bookings.")
	flag.StringVar(&namespace, "namespace", env("OPEN_RL_WORKER_NAMESPACE", "default"), "Namespace holding workers, claims and pods.")
	flag.StringVar(&deviceClass, "device-class", env("OPEN_RL_DEVICE_CLASS", "gpu.nvidia.com"), "DeviceClass generated claims request.")
	flag.StringVar(&deviceDriver, "device-driver", env("OPEN_RL_DEVICE_DRIVER", ""), "Driver publishing the ResourceSlices. Defaults to the device class.")
	flag.StringVar(&trainerTemplate, "trainer-pod-template", env("OPEN_RL_TRAINER_POD_TEMPLATE_CONFIGMAP", ""), "ConfigMap holding the default trainer pod template.")
	flag.StringVar(&samplerTemplate, "sampler-pod-template", env("OPEN_RL_SAMPLER_POD_TEMPLATE_CONFIGMAP", ""), "ConfigMap holding the default sampler pod template.")
	flag.DurationVar(&retryInterval, "retry-interval", envDuration("OPEN_RL_RECONCILE_INTERVAL", 10*time.Second), "How often an unplaced worker is retried.")
	flag.DurationVar(&placementTimeout, "placement-timeout", envDuration("OPEN_RL_PLACEMENT_TIMEOUT", 15*time.Minute),
		"How long a worker may go unplaced before the request is declared unsatisfiable. 0 waits forever.")
	flag.DurationVar(&reclaimInterval, "reclaim-interval", envDuration("OPEN_RL_RECLAIM_INTERVAL", time.Minute), "How often idle claims are swept.")

	opts := zap.Options{Development: false}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))
	setupLog := ctrl.Log.WithName("setup")

	if deviceDriver == "" {
		deviceDriver = deviceClass
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
			},
		},
	})
	if err != nil {
		setupLog.Error(err, "cannot start manager")
		os.Exit(1)
	}

	reconciler := &controller.OpenRLWorkerReconciler{
		Client:       mgr.GetClient(),
		Recorder:     mgr.GetEventRecorderFor("placement-controller"),
		Namespace:    namespace,
		DeviceClass:  deviceClass,
		DeviceDriver: deviceDriver,
		DefaultPodTemplates: map[openrlv1alpha1.WorkerRole]string{
			openrlv1alpha1.RoleTrainer: trainerTemplate,
			openrlv1alpha1.RoleSampler: samplerTemplate,
		},
		RetryInterval:    retryInterval,
		PlacementTimeout: placementTimeout,
		ReclaimInterval:  reclaimInterval,
	}
	if err := reconciler.SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "cannot set up the OpenRLWorker controller")
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

	setupLog.Info("placing workers", "namespace", namespace, "deviceClass", deviceClass, "deviceDriver", deviceDriver)
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
