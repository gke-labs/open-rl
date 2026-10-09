IMG ?= ghcr.io/gke-labs/open-rl/scheduler:latest
CONTROLLER_TOOLS_VERSION ?= v0.19.0

CRD_MANIFEST := ../deploy/base/00-workload-crd.yaml
CRD_GROUP_MANIFEST := ../deploy/base/00-claimledger-crd.yaml
LOCALBIN := $(shell pwd)/bin
CONTROLLER_GEN := $(LOCALBIN)/controller-gen
GEN_DIR := $(LOCALBIN)/generated

.PHONY: all
all: generate manifests fmt vet test build

$(LOCALBIN):
	mkdir -p $(LOCALBIN)

$(CONTROLLER_GEN): $(LOCALBIN)
	GOBIN=$(LOCALBIN) go install sigs.k8s.io/controller-tools/cmd/controller-gen@$(CONTROLLER_TOOLS_VERSION)

# api/v1alpha1/zz_generated.deepcopy.go. Checked in, so a plain `go build`
# works without the toolchain.
.PHONY: generate
generate: $(CONTROLLER_GEN)
	$(CONTROLLER_GEN) object paths="./api/..."

# The CRD schema, from the markers on the api/ types. kubebuilder would write
# this to config/crd/bases; there is no config/ tree here (see PROJECT), so it
# lands directly on the manifest scheduler/deploy applies. The types' doc comments
# become the schema descriptions, so the prose lives in Go, not in the YAML.
.PHONY: manifests
manifests: $(CONTROLLER_GEN)
	$(CONTROLLER_GEN) crd:generateEmbeddedObjectMeta=true paths="./api/..." output:crd:dir=$(GEN_DIR)/crd
	cp $(GEN_DIR)/crd/openrl.io_workloads.yaml $(CRD_MANIFEST)
	cp $(GEN_DIR)/crd/openrl.io_claimledgers.yaml $(CRD_GROUP_MANIFEST)

# Fails when the checked-in CRD no longer matches the types. For CI, and for
# the case where the generator cannot be run on the machine holding the diff.
.PHONY: manifests-check
manifests-check: $(CONTROLLER_GEN)
	$(CONTROLLER_GEN) crd:generateEmbeddedObjectMeta=true paths="./api/..." output:crd:dir=$(GEN_DIR)/crd
	diff -u $(CRD_MANIFEST) $(GEN_DIR)/crd/openrl.io_workloads.yaml
	diff -u $(CRD_GROUP_MANIFEST) $(GEN_DIR)/crd/openrl.io_claimledgers.yaml

# RBAC stays hand-written and this target only checks it. controller-gen emits
# one ClusterRole; the deployed policy is deliberately split, cluster-scoped for
# nodes and ResourceSlices and namespaced for everything else, so the controller
# cannot reach pods or claims outside its own namespace. Compare the verb sets.
.PHONY: rbac
rbac: $(CONTROLLER_GEN)
	$(CONTROLLER_GEN) rbac:roleName=open-rl-scheduler paths="./internal/..." output:rbac:dir=$(GEN_DIR)/rbac
	@echo "generated to $(GEN_DIR)/rbac; reconcile by hand with scheduler/deploy/base/01-scheduler.yaml"

.PHONY: fmt
fmt:
	go fmt ./...

# fmt without mutation, for CI: fails listing the files gofmt would rewrite.
.PHONY: fmt-check
fmt-check:
	@files="$$(gofmt -l .)"; if [ -n "$$files" ]; then echo "$$files"; exit 1; fi

.PHONY: vet
vet:
	go vet ./...

.PHONY: test
test:
	go test ./... -race -count=1

# The placement storm (internal/controller/stress_test.go). Minutes under
# -race, so it is behind a build tag and not part of `test`.
.PHONY: stress
stress:
	go test ./internal/controller/ -tags stress -run TestPlacementStorm -count=1 -timeout 30m

.PHONY: build
build:
	go build -o bin/manager ./cmd/manager

.PHONY: run
run:
	go run ./cmd/manager --leader-elect=false --namespace=$${OPEN_RL_WORKER_NAMESPACE:-default}

# The pipeline against a real API server and fake GPUs: kind plus the DRA
# example driver. No hardware needed. See hack/kind-smoke.sh for the knobs.
.PHONY: smoke
smoke:
	./hack/kind-smoke.sh

.PHONY: docker-build
docker-build:
	docker build -t $(IMG) .

.PHONY: docker-push
docker-push:
	docker push $(IMG)

.PHONY: deploy
deploy:
	kubectl apply --server-side -k ../deploy/base

.PHONY: clean
clean:
	rm -rf bin
