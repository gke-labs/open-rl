# This file contains the HTML/CSS/JS template for the Light Theme Admin Dashboard.

ADMIN_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Open-RL Admin Dashboard — Accelerator Usage</title>
  <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap">
  <style>
    :root {
      --bg-main: #f8fafc;
      --bg-card: #ffffff;
      --border-card: #e2e8f0;
      --text-main: #334155;
      --text-muted: #64748b;
      --text-heading: #0f172a;

      --role-sampler-bg: #e0f2fe;
      --role-sampler-border: #38bdf8;
      --role-sampler-text: #0369a1;

      --role-trainer-bg: #ffedd5;
      --role-trainer-border: #fb923c;
      --role-trainer-text: #c2410c;

      --idle-bg: #f1f5f9;
      --idle-border: #cbd5e1;
      --idle-text: #64748b;

      --accent-green: #10b981;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background-color: var(--bg-main);
      color: var(--text-main);
      padding: 24px;
      line-height: 1.5;
    }

    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border-card);
    }
    .header h1 {
      font-size: 22px;
      font-weight: 700;
      color: var(--text-heading);
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .header .live-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #dcfce7;
      color: #15803d;
      font-size: 12px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 9999px;
      border: 1px solid #bbf7d0;
    }
    .live-dot {
      width: 8px;
      height: 8px;
      background-color: #22c55e;
      border-radius: 50%;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(1.2); }
      100% { opacity: 1; transform: scale(1); }
    }

    .nav-tab {
      background: none;
      border: none;
      border-bottom: 2px solid transparent;
      padding: 12px 18px;
      font-size: 14px;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s ease;
    }
    .nav-tab:hover {
      color: var(--text-heading);
    }
    .nav-tab.active {
      color: #2563eb;
      border-bottom-color: #2563eb;
    }
    .sort-btn {
      background: #f8fafc;
      color: #64748b;
      font-weight: 500;
      transition: all 0.15s ease;
    }
    .sort-btn.active {
      background: #0284c7 !important;
      color: white !important;
      font-weight: 600 !important;
    }
    .badge-coming-soon {
      background: #eff6ff;
      color: #2563eb;
      border: 1px solid #bfdbfe;
      font-size: 10px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 9999px;
      text-transform: uppercase;
    }

    .apexcharts-tooltip {
      pointer-events: none !important;
      max-height: 220px !important;
      overflow-y: auto !important;
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
      border-radius: 8px !important;
    }

    #fleet-tab-root {
      width: 100%;
      box-sizing: border-box;
    }

    .cards-container {
      display: flex;
      flex-direction: column;
      gap: 24px;
      width: 100%;
    }

    .claim-card {
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    .card-title {
      font-size: 16px;
      font-weight: 700;
      color: var(--text-heading);
    }
    .card-subtitle {
      font-size: 13px;
      color: var(--text-muted);
    }

    .stats-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }
    .stat-pill {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 12px;
    }
    .stat-label {
      font-size: 12px;
      font-weight: 500;
      color: var(--text-muted);
      margin-bottom: 4px;
    }
    .stat-value {
      font-size: 18px;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-heading);
    }

    .breakdown-section {
      margin-bottom: 20px;
    }
    .breakdown-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-heading);
      margin-bottom: 8px;
    }
    .progress-bar-wrap {
      height: 16px;
      width: 100%;
      background: #f1f5f9;
      border-radius: 9999px;
      overflow: hidden;
      display: flex;
      border: 1px solid #e2e8f0;
    }
    .progress-seg {
      height: 100%;
      transition: width 0.3s ease;
    }
    .breakdown-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-top: 8px;
      font-size: 12px;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 3px;
    }

    .chart-controls {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      background: #f1f5f9;
      padding: 4px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
    }
    .toggle-btn {
      padding: 4px 10px;
      border-radius: 4px;
      border: none;
      background: transparent;
      color: var(--text-muted);
      cursor: pointer;
    }
    .toggle-btn.active {
      background: #ffffff;
      color: var(--text-heading);
      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    .chart-area {
      min-height: 280px;
    }

    /* Option 2: Visual Block Grid System */
    .req-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 8px;
      margin-bottom: 12px;
    }
    .req-block {
      width: 14px;
      height: 14px;
      border-radius: 3px;
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
      position: relative;
    }
    .req-block:hover {
      transform: scale(1.4);
      z-index: 10;
      box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .req-block.executing {
      background-color: #22c55e;
      border: 1px solid #16a34a;
    }
    .req-block.pending-trainer { background-color: #f97316; border: 1px solid #ea580c; }
    .req-block.pending-sampler { background-color: #0284c7; border: 1px solid #0369a1; }
    .req-block.done-sampler { background-color: #7dd3fc; border: 1px solid #0284c7; }
    .req-block.done-trainer { background-color: #ffedd5; border: 1px solid #f97316; }
    .req-block.failed { background-color: #ef4444; border: 1px solid #dc2626; }

    .step-box {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .step-title {
      font-size: 14px;
      font-weight: 700;
      color: #0f172a;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid #f1f5f9;
    }
    .phase-title {
      font-size: 12px;
      font-weight: 600;
      color: #475569;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 8px;
    }

    .job-row {
      cursor: pointer;
      transition: background 0.15s ease;
    }
    .job-row:hover {
      background-color: #f8fafc !important;
    }
  </style>
</head>
<body>

  <div class="header">
    <h1 style="font-size: 20px; font-weight: 700; color: #0f172a; margin: 0;">OpenRL Admin Dashboard</h1>
  </div>

  <div class="tabs" style="border-bottom: 2px solid #e2e8f0; margin-bottom: 24px; display: flex; gap: 4px;">
    <button class="nav-tab active" id="tab-accel" onclick="switchNavTab('accel')">Fleet</button>
    <button class="nav-tab" id="tab-jobs" onclick="switchNavTab('jobs')">RL Jobs</button>
    <button class="nav-tab" id="tab-job-details" style="display: none;" onclick="switchNavTab('job-details')">
      Job Details
    </button>
  </div>

  <div id="fleet-tab-root">
    <div
      style="display: flex; align-items: center; justify-content: flex-end; gap: 16px; "
      style="margin-bottom: 16px; flex-wrap: wrap; font-size: 13px; color: var(--text-muted);"
    >
      <div style="display: flex; align-items: center; gap: 8px;">
        <div class="toggle-group">
          <button class="toggle-btn view-btn active" data-view="timeline" onclick="setGlobalView('timeline')">📊 Timeline</button>
          <button class="toggle-btn view-btn" data-view="breakdown" onclick="setGlobalView('breakdown')">⏱️ Usage Breakdown</button>
        </div>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <span>Time:</span>
        <select
          id="window-select"
          onchange="setWindow(parseInt(this.value))"
          style="padding: 4px 10px; border-radius: 6px; border: 1px solid #cbd5e1;
                 font-size: 12px; background: white; font-weight: 500;
                 color: #334155; cursor: pointer;"
        >
          <option value="60">1 min</option>
          <option value="300" selected>5 min</option>
          <option value="900">15 min</option>
          <option value="3600">1 hour</option>
          <option value="10800">3 hours</option>
          <option value="21600">6 hours</option>
          <option value="86400">24 hours</option>
          <option value="0">All History</option>
          <option value="-1">📅 Custom Range...</option>
        </select>
        <div id="custom-time-inputs" style="display: none; align-items: center; gap: 6px;">
          <input
            type="datetime-local"
            id="custom-start-time"
            style="padding: 3px 6px; border-radius: 6px; border: 1px solid #cbd5e1; font-size: 12px;"
          />
          <span style="color: #64748b; font-size: 12px;">to</span>
          <input
            type="datetime-local"
            id="custom-end-time"
            style="padding: 3px 6px; border-radius: 6px; border: 1px solid #cbd5e1; font-size: 12px;"
          />
          <button
            onclick="applyCustomTimeRange()"
            style="background: #0284c7; color: white; border: none;
                   border-radius: 6px; padding: 4px 10px; font-size: 12px;
                   font-weight: 600; cursor: pointer;"
          >Apply</button>
        </div>
      </div>
    </div>
    <div class="cards-container" id="cards-root">
      <!-- Dynamic DRA ResourceClaim cards injected here -->
    </div>
  </div>

    <div id="jobs-root" style="display: none; background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
      <div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 16px;">
        <div style="display: flex; gap: 8px;">
          <button class="toggle-btn active" id="btn-all-jobs" onclick="filterJobs('all')">All Jobs</button>
          <button class="toggle-btn" id="btn-active-jobs" onclick="filterJobs('active')">Active Jobs</button>
          <button class="toggle-btn" id="btn-completed-jobs" onclick="filterJobs('completed')">Completed Jobs</button>
        </div>
      </div>

      <!-- Active Jobs Table -->
      <div style="overflow-x: auto; margin-bottom: 16px;">
        <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left;">
          <thead>
            <tr style="border-bottom: 2px solid #e2e8f0; color: #64748b;">
              <th style="padding: 10px;">Job / Model ID</th>
              <th style="padding: 10px;">Base Model</th>
              <th style="padding: 10px;">Tenant</th>
              <th style="padding: 10px;">Status</th>
              <th style="padding: 10px;">Step Progress</th>
              <th style="padding: 10px;">Pending Queues</th>
              <th style="padding: 10px;">Created At</th>
              <th style="padding: 10px;">Updated At</th>
              <th style="padding: 10px;">Action</th>
            </tr>
          </thead>
          <tbody id="jobs-table-body">
            <tr><td colspan="9" style="padding: 20px; text-align: center; color: #94a3b8;">Loading cluster jobs...</td></tr>
          </tbody>
        </table>
      </div>

    </div>

    <!-- Dedicated Full-Page Job Details View -->
    <div
      id="job-details-root"
      style="display: none; background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; margin-bottom: 24px;"
    >
      <div style="margin-bottom: 16px;">
        <button
          onclick="backToJobs()"
          style="background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1;
                 border-radius: 8px; padding: 8px 16px; font-weight: 600;
                 font-size: 13px; font-family: inherit; cursor: pointer;
                 display: inline-flex; align-items: center; gap: 6px;
                 box-shadow: 0 1px 2px rgba(0,0,0,0.05);"
        >← Back to Jobs</button>
      </div>

      <!-- Job Metadata Summary Card Banner -->
      <div
        id="job-metadata-banner"
        style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
               padding: 16px 20px; margin-bottom: 24px; display: grid;
               grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px;"
      >
        <div>
          <div style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 4px;">Job / Model ID</div>
          <div id="meta-job-id" style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #0284c7; font-size: 13px;">-</div>
        </div>
        <div>
          <div style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 4px;">Base Model</div>
          <div id="meta-base-model" style="font-weight: 600; color: #0f172a; font-size: 13px;">-</div>
        </div>
        <div>
          <div style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 4px;">Kind / Tenant</div>
          <div id="meta-kind-tenant" style="font-weight: 500; color: #334155; font-size: 13px;">-</div>
        </div>
        <div>
          <div style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 4px;">Launched At</div>
          <div id="meta-launched-at" style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #334155;">-</div>
        </div>
        <div>
          <div style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 4px;">Status / Step</div>
          <div id="meta-status-step" style="font-weight: 600; font-size: 13px;">-</div>
        </div>
      </div>

      <!-- Visual Block Status Legend & Counts Cards -->
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px;">
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px 14px;">
          <div style="display: flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 600; color: #166534; text-transform: uppercase;">
            <div class="req-block executing"></div> Executing
          </div>
          <div id="count-exec" style="font-size: 18px; font-weight: 700; color: #15803d; margin-top: 4px; line-height: 1.2;">0</div>
        </div>

        <div style="background: #fff7ed; border: 1px solid #ffedd5; border-radius: 8px; padding: 10px 14px;">
          <div style="display: flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 600; color: #9a3412; text-transform: uppercase;">
            <div class="req-block pending-trainer"></div> Train Requests Pending
          </div>
          <div id="count-tr-pend" style="font-size: 18px; font-weight: 700; color: #c2410c; margin-top: 4px; line-height: 1.2;">0</div>
        </div>

        <div style="background: #f0f9ff; border: 1px solid #e0f2fe; border-radius: 8px; padding: 10px 14px;">
          <div style="display: flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 600; color: #075985; text-transform: uppercase;">
            <div class="req-block pending-sampler"></div> Sampling Requests Pending
          </div>
          <div id="count-sa-pend" style="font-size: 18px; font-weight: 700; color: #0369a1; margin-top: 4px; line-height: 1.2;">0</div>
        </div>

        <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 10px 14px;">
          <div style="display: flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 600; color: #0369a1; text-transform: uppercase;">
            <div class="req-block done-sampler"></div> Sampling Requests Done
          </div>
          <div id="count-sa-done" style="font-size: 18px; font-weight: 700; color: #0284c7; margin-top: 4px; line-height: 1.2;">0</div>
        </div>

        <div style="background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 10px 14px;">
          <div style="display: flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 600; color: #c2410c; text-transform: uppercase;">
            <div class="req-block done-trainer"></div> Training Requests Done
          </div>
          <div id="count-tr-done" style="font-size: 18px; font-weight: 700; color: #ea580c; margin-top: 4px; line-height: 1.2;">0</div>
        </div>
      </div>

      <!-- Parameter Mutation Progression Line Chart Card -->
      <div id="mutation-chart-card"
           style="display: none; background: white; border: 1px solid #e2e8f0; border-radius: 12px; ` +
             `padding: 28px 32px 24px 32px; margin-top: 28px; margin-bottom: 28px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; ` +
             `padding: 4px 6px 14px 6px; border-bottom: 1px solid #f1f5f9;">
          <div style="padding-left: 2px;">
            <h3 style="font-size: 15px; font-weight: 700; color: #0f172a; display: flex; align-items: center; gap: 8px; margin: 0;">
              <span>⚡</span> Parameter Mutation Progression per Step
            </h3>
            <div style="font-size: 12px; color: #64748b; margin-top: 4px; padding-left: 2px;">
              Percentage of model weights mutated across training steps via sparse delta weight synchronization
            </div>
          </div>
          <div id="mutation-summary-badge"
               style="padding: 6px 14px; border-radius: 9999px; font-size: 11px; ` +
                 `font-weight: 600; background: #fff7ed; color: #c2410c; border: 1px solid #ffedd5;">
            Delta Sync Active
          </div>
        </div>
        <div id="mutation-chart-root" style="min-height: 250px; padding: 4px 8px;"></div>
      </div>

      <!-- Step Matrix Header Bar -->
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; ` +
           `padding: 0 2px;">
        <div style="font-weight: 700; font-size: 15px; color: #0f172a; display: flex; align-items: center; gap: 8px;">
          <span>📌</span> Step-Based Execution Matrix
        </div>
        <!-- Role Filter -->
        <div style="display: flex; align-items: center; gap: 6px; font-size: 12px; color: #64748b;">
          <span style="font-weight: 500;">Filter:</span>
          <select id="role-filter" onchange="renderJobInspection(cachedJobRequestsData)" ` +
            `style="padding: 5px 10px; border-radius: 6px; border: 1px solid #cbd5e1; font-size: 12px; ` +
            `background: white; color: #334155; font-family: inherit;">
            <option value="all">All Requests</option>
            <option value="sampler">Sampler Requests Only</option>
            <option value="trainer">Trainer Requests Only</option>
          </select>
        </div>
      </div>

      <!-- Step-Based Matrix Grid Root -->
      <div id="step-feed-root">
        <div style="color: #94a3b8;">Loading job requests grid...</div>
      </div>
    </div>
  </div>

  <script>
    let activeMetric = 'tokens'; // 'tokens'
    let activeWindowSec = 300; // default 5 minutes
    let activeGlobalView = 'timeline'; // default 'timeline'
    let chartInstances = {};

    function switchNavTab(tabName) {
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      const activeTab = document.getElementById(`tab-${tabName}`);
      if (activeTab) {
        activeTab.style.display = 'inline-flex';
        activeTab.classList.add('active');
      }

      const accelSection = document.getElementById('fleet-tab-root') || document.getElementById('cards-root');
      const jobsSection = document.getElementById('jobs-root');
      const jobDetailsSection = document.getElementById('job-details-root');

      if (tabName === 'accel') {
        if (accelSection) accelSection.style.display = 'block';
        if (jobsSection) jobsSection.style.display = 'none';
        if (jobDetailsSection) jobDetailsSection.style.display = 'none';
      } else if (tabName === 'jobs') {
        if (accelSection) accelSection.style.display = 'none';
        if (jobsSection) jobsSection.style.display = 'block';
        if (jobDetailsSection) jobDetailsSection.style.display = 'none';
      } else if (tabName === 'job-details') {
        if (accelSection) accelSection.style.display = 'none';
        if (jobsSection) jobsSection.style.display = 'none';
        if (jobDetailsSection) jobDetailsSection.style.display = 'block';
      }
    }

    function inspectJob(jobId) {
      selectedJobId = jobId;

      let jobMeta = null;
      if (cachedJobsData) {
        const allJobs = [...(cachedJobsData.active_jobs || []), ...(cachedJobsData.completed_jobs || [])];
        jobMeta = allJobs.find(j => j.model_id === jobId);
      }

      const mIdEl = document.getElementById('meta-job-id');
      if (mIdEl) mIdEl.innerText = jobId;
      const mModelEl = document.getElementById('meta-base-model');
      if (mModelEl) mModelEl.innerText = jobMeta ? jobMeta.base_model : 'Qwen/Qwen3-8B';
      const mKindEl = document.getElementById('meta-kind-tenant');
      if (mKindEl) {
        let kindStr = jobMeta ? `${jobMeta.training_kind.toUpperCase()} (${jobMeta.tenant_id})` : 'FFT (default)';
        if (jobMeta && jobMeta.weight_sync_strategy === 'delta') {
          const mutStr = (jobMeta.latest_mutation_pct !== undefined && jobMeta.latest_mutation_pct !== null)
            ? `${jobMeta.latest_mutation_pct}% mutated`
            : 'delta';
          kindStr += ` │ ⚡ Delta Sync (${mutStr})`;
        }
        mKindEl.innerText = kindStr;
      }
      const mLaunchEl = document.getElementById('meta-launched-at');
      if (mLaunchEl) mLaunchEl.innerText = jobMeta ? formatTs(jobMeta.created_at) : '-';
      const mStatusEl = document.getElementById('meta-status-step');
      if (mStatusEl) {
        const stText = jobMeta ? (jobMeta.status === 'active' ? '🟢 ACTIVE' : '⚪ COMPLETED') : 'COMPLETED';
        const stepText = jobMeta ? ` │ Step ${jobMeta.current_step}` : '';
        mStatusEl.innerHTML = `<span style="color:${jobMeta && jobMeta.status === 'active' ? '#15803d' : '#475569'};">${stText}</span>${stepText}`;
      }

      const tabBtn = document.getElementById('tab-job-details');
      if (tabBtn) {
        tabBtn.innerText = `Job: ${jobId.substring(0, 10)}...`;
        tabBtn.style.display = 'inline-flex';
      }
      switchNavTab('job-details');
      fetchJobRequests(jobId);
    }

    function backToJobs() {
      selectedJobId = null;
      const tabBtn = document.getElementById('tab-job-details');
      if (tabBtn) tabBtn.style.display = 'none';
      switchNavTab('jobs');
    }

    async function fetchJobRequests(jobId) {
      try {
        const resp = await fetch(`/api/v1/admin/jobs/${jobId}/requests`);
        if (!resp.ok) return;
        const data = await resp.json();
        cachedJobRequestsData = data;
        renderJobInspection(data);
      } catch (e) {
        console.error('Failed to fetch job requests', e);
      }
    }

    function renderJobInspection(data) {
      // 1. Currently Executing Request Card
      const execEl = document.getElementById('executing-content');
      if (execEl) {
        if (data.currently_executing) {
          const ex = data.currently_executing;
          const roleColor = ex.role === 'sampler' ? '#0284c7' : '#c2410c';
          execEl.innerHTML = `
            <div><strong style="color: ${roleColor};">[${ex.role.toUpperCase()}]</strong> Request ID: ` +
            `<span style="font-weight:600;">${ex.request_id}</span></div>
            <div style="margin-top:4px; font-size:12px; color:#475569;">Operation: <strong>${ex.op}</strong> │ ` +
            `Pod: <strong>${ex.worker_pod}</strong> │ Running: <strong style="color:#d97706;">${ex.elapsed_sec}s</strong></div>
          `;
        } else {
          execEl.innerHTML = `<span style="color: #94a3b8;">No request currently executing on GPU</span>`;
        }
      }

      // 2. Visual Block Status Legend & Counts Breakdown
      const trPend = (data.pending_queues.trainer || []).length;
      const saPend = (data.pending_queues.sampler || []).length;
      const recents = data.recent_completed || [];
      const saDone = recents.filter(r => r.role === 'sampler').length;
      const trDone = recents.filter(r => r.role === 'trainer').length;
      const isExec = data.currently_executing ? 1 : 0;

      const cExec = document.getElementById('count-exec');
      if (cExec) cExec.innerText = isExec;
      const cTrPend = document.getElementById('count-tr-pend');
      if (cTrPend) cTrPend.innerText = trPend;
      const cSaPend = document.getElementById('count-sa-pend');
      if (cSaPend) cSaPend.innerText = saPend;
      const cSaDone = document.getElementById('count-sa-done');
      if (cSaDone) cSaDone.innerText = saDone;
      const cTrDone = document.getElementById('count-tr-done');
      if (cTrDone) cTrDone.innerText = trDone;

      // 3. Render Parameter Mutation Progression Bar Chart (if delta sync)
      renderMutationChart(data.mutation_history, data.weight_sync_strategy);

      // 3. 3-Tier Step-Centric Reverse Chronological Timeline Grid
      const feedEl = document.getElementById('step-feed-root');
      const roleFilter = document.getElementById('role-filter')?.value || 'all';

      // Combine active + pending + completed requests
      const rawReqs = [];
      if (data.currently_executing) {
        rawReqs.push({ ...data.currently_executing, state: 'executing' });
      }
      (data.pending_queues.trainer || []).forEach(r => rawReqs.push({ ...r, state: 'pending-trainer' }));
      (data.pending_queues.sampler || []).forEach(r => rawReqs.push({ ...r, state: 'pending-sampler' }));
      (data.recent_completed || []).forEach(r => rawReqs.push({ ...r, state: r.role === 'sampler' ? 'done-sampler' : 'done-trainer' }));

      // Filter out sentinel noise (create_model, SHUTDOWN_SENTINEL)
      let filtered = rawReqs.filter(
        r => r.op !== 'create_model' && r.op !== 'create_model_from_state' && r.request_id !== 'SHUTDOWN_SENTINEL'
      );

      if (roleFilter !== 'all') {
        filtered = filtered.filter(r => r.role === roleFilter);
      }

      if (filtered.length === 0) {
        feedEl.innerHTML = '<span style="color:#94a3b8;">No request blocks found matching filter.</span>';
        return;
      }

      // Step Grouping Logic
      const stepMap = {};
      const sortedReqs = [...filtered].sort(
        (a, b) => (a.created_at || a.started_at || a.completed_at || 0) - (b.created_at || b.started_at || b.completed_at || 0)
      );

      let currentStepIdx = 0;
      sortedReqs.forEach(r => {
        let sKey = null;
        let sName = null;
        let sOrder = 0;

        if (r.session_id) {
          const match = r.session_id.match(/(?:sampler|step)[-_]?(\d+)/i);
          if (match) {
            const val = parseInt(match[1]);
            if (val > 10000) {
              sKey = 'step_0';
              sName = 'Step 0 (Initial Rollout)';
              sOrder = 0;
            } else {
              currentStepIdx = val;
              sKey = `step_${val}`;
              sName = `Step ${val}`;
              sOrder = val;
            }
          }
        }

        if (!sKey) {
          if (currentStepIdx === 0) {
            sKey = 'step_0';
            sName = 'Step 0 (Initial Rollout)';
            sOrder = 0;
          } else {
            sKey = `step_${currentStepIdx}`;
            sName = `Step ${currentStepIdx}`;
            sOrder = currentStepIdx;
          }
        }

        if (r.op === 'optim_step') {
          currentStepIdx = Math.max(currentStepIdx + 1, sOrder + 1);
        }

        if (!stepMap[sKey]) {
          stepMap[sKey] = { key: sKey, name: sName, order: sOrder, sampler: [], trainer: [] };
        }

        if (r.role === 'sampler') {
          stepMap[sKey].sampler.push(r);
        } else {
          stepMap[sKey].trainer.push(r);
        }
      });

      // Sort step cards reverse-chronologically (Step N -> Step 0)
      const stepList = Object.values(stepMap).sort((a, b) => b.order - a.order);

      feedEl.innerHTML = stepList.map(st => {
        const samplerBlocks = st.sampler;
        const trainerBlocks = st.trainer;
        const totalTokens = samplerBlocks.reduce((acc, r) => acc + (r.token_count || 0), 0);
        const activeBadge = (st.order === currentStepIdx && data.currently_executing) ?
          `<span style="background:#dcfce7; color:#15803d; padding:2px 8px; ` +
          `border-radius:9999px; font-size:11px; font-weight:600; margin-left:8px;">🟢 IN PROGRESS</span>` : '';

        const samplerHtml = samplerBlocks.map(r => {
          const cls = r.state === 'executing' ? 'executing' : (r.state || 'done-sampler');
          const tokInfo = r.token_count ? ` | Tokens: ${r.token_count}` : '';
          const title = `Req: ${r.request_id} | Op: ${r.op}${tokInfo} | Status: ${r.status || r.state}`;
          return `<div class="req-block ${cls}" title="${title}"></div>`;
        }).join('');

        const trainerHtml = trainerBlocks.map(r => {
          const cls = r.state === 'executing' ? 'executing' : (r.state || 'done-trainer');
          const title = `Req: ${r.request_id} | Op: ${r.op} | Status: ${r.status || r.state}`;
          return `<div class="req-block ${cls}" title="${title}"></div>`;
        }).join('');

        const samplerSummary = totalTokens > 0 ?
          `${samplerBlocks.length} Trajectories (${totalTokens.toLocaleString()} tokens)` :
          (samplerBlocks.length > 0 ? `${samplerBlocks.length} Trajectories` : '0 Rollout Requests');

        const noSamplerText = st.order > 0 && samplerBlocks.length === 0 ?
          '<span style="color:#94a3b8; font-size:12px;">Final Training & Checkpoint Step (No rollouts after step completion)</span>' :
          '<span style="color:#94a3b8; font-size:12px;">No sampler blocks</span>';

        return `
          <div class="step-box">
            <div class="step-title">
              <div>
                <span>${st.name}</span>
                ${activeBadge}
              </div>
              <span style="font-size:12px; color:#64748b; font-weight:500;">
                ${samplerBlocks.length + trainerBlocks.length} total requests
              </span>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;">
              <div style="background: #f8fafc; border: 1px solid #f1f5f9; border-radius: 8px; padding: 12px;">
                <div class="phase-title" style="margin-top: 0;">
                  <span>🔹 Rollout Phase (vLLM Sampler)</span>
                  <span style="color:#0284c7;">${samplerSummary}</span>
                </div>
                <div class="req-grid" style="margin-top: 8px;">
                  ${samplerHtml.length > 0 ? samplerHtml : noSamplerText}
                </div>
              </div>

              <div style="background: #f8fafc; border: 1px solid #f1f5f9; border-radius: 8px; padding: 12px;">
                <div class="phase-title" style="margin-top: 0;">
                  <span>🔸 Training Phase (PyTorch Trainer)</span>
                  <span style="color:#ea580c;">${trainerBlocks.length} Trainer Requests</span>
                </div>
                <div class="req-grid" style="margin-top: 8px;">
                  ${trainerHtml.length > 0 ? trainerHtml : '<span style="color:#94a3b8; font-size:12px;">No trainer blocks</span>'}
                </div>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    function setGlobalView(viewName) {
      activeGlobalView = viewName;
      document.querySelectorAll('.view-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === viewName);
      });
      document.querySelectorAll('.panel-timeline').forEach(panel => {
        panel.style.display = viewName === 'timeline' ? 'block' : 'none';
      });
      document.querySelectorAll('.panel-breakdown').forEach(panel => {
        panel.style.display = viewName === 'breakdown' ? 'block' : 'none';
      });
    }

    let customStartTs = null;
    let customEndTs = null;

    function setWindow(sec) {
      if (sec === -1) {
        const customDiv = document.getElementById('custom-time-inputs');
        if (customDiv) customDiv.style.display = 'inline-flex';
        return;
      } else {
        const customDiv = document.getElementById('custom-time-inputs');
        if (customDiv) customDiv.style.display = 'none';
        customStartTs = null;
        customEndTs = null;
      }
      activeWindowSec = sec;
      const select = document.getElementById('window-select');
      if (select) select.value = sec;
      fetchAndUpdate();
    }

    function applyCustomTimeRange() {
      const startVal = document.getElementById('custom-start-time')?.value;
      const endVal = document.getElementById('custom-end-time')?.value;
      if (!startVal || !endVal) return;

      const startMs = new Date(startVal).getTime();
      const endMs = new Date(endVal).getTime();
      if (isNaN(startMs) || isNaN(endMs) || endMs <= startMs) return;

      customStartTs = startMs / 1000.0;
      customEndTs = endMs / 1000.0;
      fetchAndUpdate();
    }

    function onTimelineDragZoom(minMs, maxMs) {
      if (!minMs || !maxMs || maxMs <= minMs) return;

      const startDate = new Date(minMs);
      const endDate = new Date(maxMs);
      const pad = (n) => String(n).padStart(2, '0');
      const formatLocal = (d) =>
        `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;

      const startInput = document.getElementById('custom-start-time');
      const endInput = document.getElementById('custom-end-time');
      const select = document.getElementById('window-select');
      const customDiv = document.getElementById('custom-time-inputs');

      if (startInput) startInput.value = formatLocal(startDate);
      if (endInput) endInput.value = formatLocal(endDate);
      if (select) select.value = '-1';
      if (customDiv) customDiv.style.display = 'inline-flex';

      customStartTs = minMs / 1000.0;
      customEndTs = maxMs / 1000.0;
      fetchAndUpdate();
    }

    function toggleMetric(metric) {
      activeMetric = metric;
      document.querySelectorAll('.toggle-btn[data-metric]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.metric === metric);
      });
      fetchAndUpdate();
    }

    async function fetchAndUpdate() {
      try {
        let url = `/api/v1/admin/accel_usage?window_sec=${activeWindowSec}`;
        if (customStartTs && customEndTs) {
          url = `/api/v1/admin/accel_usage?start_ts=${customStartTs}&end_ts=${customEndTs}`;
        }
        const res = await fetch(url);
        const data = await res.json();
        renderDashboard(data.claims || {});
      } catch (err) {
        console.error('Failed to fetch accel usage:', err);
      }
    }

    const GLOBAL_PALETTE = ['#0284c7', '#ea580c', '#16a34a', '#9333ea', '#db2777', '#0891b2', '#d97706', '#4f46e5'];
    const tenantColorMap = { 'Idle': '#e2e8f0' };
    let colorIndex = 0;

    function getTenantColor(tenantId) {
      if (tenantId === 'Idle') return '#e2e8f0';
      if (!tenantColorMap[tenantId]) {
        tenantColorMap[tenantId] = GLOBAL_PALETTE[colorIndex % GLOBAL_PALETTE.length];
        colorIndex++;
      }
      return tenantColorMap[tenantId];
    }

    function formatTenantId(tenantId) {
      if (!tenantId || tenantId === 'Idle') return 'Idle';
      if (cachedJobsData) {
        const allJobs = [...(cachedJobsData.active_jobs || []), ...(cachedJobsData.completed_jobs || [])];
        const found = allJobs.find(j => j.model_id === tenantId);
        if (found) {
          const shortModel = (found.base_model || 'Job').replace('Qwen/Qwen', 'Qwen').replace('google/', '');
          return `${shortModel} (${tenantId.substring(0, 8)})`;
        }
      }
      if (tenantId.length > 12) {
        return `job-${tenantId.substring(0, 8)}`;
      }
      return tenantId;
    }

    function renderDashboard(claims) {
      const root = document.getElementById('cards-root');
      const claimKeys = Object.keys(claims).sort();

      if (claimKeys.length === 0) {
        root.innerHTML = '<div style="text-align:center; padding: 40px; color: var(--text-muted);">No active DRA ResourceClaims recorded yet.</div>';
        return;
      }

      const timeMap = {};
      claimKeys.forEach(cId => {
        ((claims[cId] || {}).history || []).forEach(ev => {
          const tSec = Math.floor(ev.acquire_time);
          if (!timeMap[tSec]) timeMap[tSec] = new Date(tSec * 1000).toLocaleTimeString();
        });
      });
      let globalMinMs = null;
      let globalMaxMs = null;
      claimKeys.forEach(cId => {
        const history = (claims[cId] || {}).history || [];
        history.forEach(ev => {
          const tAcq = Math.round(ev.acquire_time * 1000);
          const tRel = Math.round((ev.release_time || (ev.acquire_time + ev.duration_ms / 1000)) * 1000);
          if (globalMinMs === null || tAcq < globalMinMs) globalMinMs = tAcq;
          if (globalMaxMs === null || tRel > globalMaxMs) globalMaxMs = tRel;
        });
      });
      if (globalMinMs !== null && globalMaxMs !== null) {
        globalMinMs = globalMinMs - 2000;
        globalMaxMs = globalMaxMs + 10000;
      }

      claimKeys.forEach(claimId => {
        const claim = claims[claimId];
        let card = document.getElementById(`claim-card-${claimId}`);
        let windowLabel = 'Last 5m';
        if (claim.window_sec === 0) {
          windowLabel = 'All History';
        } else if (claim.window_sec >= 60) {
          windowLabel = 'Last ' + Math.round(claim.window_sec / 60) + 'm';
        } else if (claim.window_sec > 0) {
          windowLabel = 'Last ' + claim.window_sec + 's';
        }

        if (!card) {
          card = document.createElement('div');
          card.id = `claim-card-${claimId}`;
          card.className = 'claim-card';
          card.innerHTML = `
            <div class="card-header">
              <div>
                <div class="card-title">Accelerator: ${claim.resource_claim_id}</div>
                <div class="card-subtitle">Hardware: ${claim.hardware_name}</div>
              </div>
            </div>
            <div class="stats-row">
              <div class="stat-pill">
                <div class="stat-label stat-window-label">GPU Duty Cycle (${windowLabel})</div>
                <div class="stat-value stat-duty" style="color: #059669;">🔥 ${claim.duty_cycle_pct}%</div>
              </div>
              <div class="stat-pill">
                <div class="stat-label">GPU Idle Time</div>
                <div class="stat-value stat-idle" style="color: #64748b;">🟩 ${claim.idle_pct}%</div>
              </div>
            </div>
            <div class="panel-timeline panel-timeline-${claimId}" style="display: block;">
              <div class="chart-area" id="chart-${claimId}"></div>
            </div>
            <div class="panel-breakdown panel-breakdown-${claimId}" style="display: none;">
              <div class="chart-area" id="pie-chart-${claimId}"></div>
            </div>
          `;
          root.appendChild(card);
        } else {
          const dutyEl = card.querySelector('.stat-duty');
          if (dutyEl) dutyEl.innerHTML = `🔥 ${claim.duty_cycle_pct}%`;
          const idleEl = card.querySelector('.stat-idle');
          if (idleEl) idleEl.innerHTML = `🟩 ${claim.idle_pct}%`;
          const windowTitleEl = card.querySelector('.stat-window-label');
          if (windowTitleEl) windowTitleEl.innerText = `GPU Duty Cycle (${windowLabel})`;
        }
        renderTimelineChart(claimId, claim.history || [], claim.tenant_breakdown || [], globalMinMs, globalMaxMs);
        renderPieChart(claimId, claim.tenant_breakdown || [], windowLabel);
      });
      setGlobalView(activeGlobalView);
    }

    const pieChartInstances = {};

    function highlightJobSeriesAcrossCharts(targetName) {
      if (!targetName || targetName === 'Idle') return;

      Object.keys(chartInstances).forEach(cId => {
        const chartEl = document.getElementById(`chart-${cId}`);
        const chart = chartInstances[cId];
        if (!chartEl || !chart || !chart.w || !chart.w.config || !chart.w.config.series) return;
        const seriesList = chart.w.config.series;

        const allSeriesEls = chartEl.querySelectorAll('.apexcharts-series');
        allSeriesEls.forEach((el, idx) => {
          const sName = seriesList[idx]?.name;
          if (sName === targetName) {
            el.style.opacity = '1';
            el.querySelectorAll('path').forEach(p => p.style.strokeWidth = '4px');
          } else {
            el.style.opacity = '0.15';
            el.querySelectorAll('path').forEach(p => p.style.strokeWidth = '1.5px');
          }
        });
      });

      Object.keys(pieChartInstances).forEach(cId => {
        const pieEl = document.getElementById(`pie-chart-${cId}`);
        const pie = pieChartInstances[cId];
        if (!pieEl || !pie || !pie.w || !pie.w.config || !pie.w.config.labels) return;
        const labels = pie.w.config.labels;

        const slices = pieEl.querySelectorAll('.apexcharts-pie-series');
        slices.forEach((slice, idx) => {
          const lName = labels[idx];
          if (lName === targetName) {
            slice.style.opacity = '1';
          } else {
            slice.style.opacity = '0.2';
          }
        });
      });
    }

    function resetJobSeriesHighlightAcrossCharts() {
      Object.keys(chartInstances).forEach(cId => {
        const chartEl = document.getElementById(`chart-${cId}`);
        if (chartEl) {
          const allSeriesEls = chartEl.querySelectorAll('.apexcharts-series');
          allSeriesEls.forEach(el => {
            el.style.opacity = '1';
            el.querySelectorAll('path').forEach(p => p.style.strokeWidth = '2.5px');
          });
        }
      });

      Object.keys(pieChartInstances).forEach(cId => {
        const pieEl = document.getElementById(`pie-chart-${cId}`);
        if (pieEl) {
          const slices = pieEl.querySelectorAll('.apexcharts-pie-series');
          slices.forEach(slice => {
            slice.style.opacity = '1';
          });
        }
      });
    }

    function renderPieChart(claimId, tenantBreakdown, windowLabel) {
      const pieEl = document.getElementById(`pie-chart-${claimId}`);
      if (!pieEl) return;
      const validBreakdown = (tenantBreakdown || []).filter(t => t.percentage > 0);
      const labels = validBreakdown.map(t => formatTenantId(t.tenant_id));
      const series = validBreakdown.map(t => parseFloat(t.percentage.toFixed(1)));
      const colors = validBreakdown.map(t => getTenantColor(t.tenant_id));
      const options = {
        chart: {
          type: 'donut',
          height: 240,
          animations: { enabled: false },
          events: {
            legendItemMouseOver: function(chartContext, seriesIndex, config) {
              const label = config.config.labels[seriesIndex];
              if (label && label !== 'Idle') {
                highlightJobSeriesAcrossCharts(label);
              }
            },
            legendItemMouseOut: function(chartContext, seriesIndex, config) {
              resetJobSeriesHighlightAcrossCharts();
            }
          }
        },
        title: { text: `Tenant Time Share (${windowLabel})`, style: { fontSize: '13px', fontWeight: '600', color: '#0f172a' } },
        labels: labels,
        series: series,
        colors: colors,
        plotOptions: {
          pie: { donut: { size: '60%', labels: { show: true, total: { show: true, label: 'GPU Share', formatter: () => '100%' } } } }
        },
        legend: { show: true, position: 'right', fontSize: '12px' },
        dataLabels: { enabled: true, formatter: (val) => val.toFixed(1) + '%' }
      };
      if (pieChartInstances[claimId]) {
        pieChartInstances[claimId].updateOptions(options, true, false);
      } else {
        pieChartInstances[claimId] = new ApexCharts(pieEl, options);
        pieChartInstances[claimId].render();
      }
    }

    function renderTimelineChart(claimId, history, tenantBreakdown, globalMinMs, globalMaxMs) {
      const chartEl = document.getElementById(`chart-${claimId}`);
      if (!chartEl) return;

      const rawEvents = (history || []);
      const uniqueTenants = Array.from(new Set(rawEvents.map(ev => ev.tenant_id)));

      let series = [];

      if (rawEvents.length > 0) {
        series = uniqueTenants.map(t_id => {
          const points = [];
          for (let i = 0; i < rawEvents.length; i++) {
            const ev = rawEvents[i];
            const isTargetTenant = (ev.tenant_id === t_id);
            const val = isTargetTenant ? 1 : 0;
            const tAcquireMs = Math.round(ev.acquire_time * 1000);
            const tReleaseMs = Math.round((ev.release_time || (ev.acquire_time + ev.duration_ms / 1000)) * 1000);

            points.push([tAcquireMs - 1, 0]);
            points.push([tAcquireMs, val]);
            points.push([tReleaseMs, val]);
            points.push([tReleaseMs + 1, 0]);
          }
          return { name: formatTenantId(t_id), data: points };
        });
      } else {
        const now = Date.now();
        series = [{ name: 'Idle', data: [[now - 60000, 0], [now, 0]] }];
      }

      const seriesColors = uniqueTenants.length > 0 ? uniqueTenants.map(t_id => getTenantColor(t_id)) : ['#64748b'];

      const options = {
        chart: {
          type: 'line',
          height: 220,
          toolbar: { show: true, autoSelected: 'zoom' },
          zoom: { enabled: true, type: 'x', autoScaleYaxis: false },
          selection: { enabled: true, type: 'x' },
          events: {
            legendItemMouseOver: function(chartContext, seriesIndex, config) {
              const seriesName = config.config.series[seriesIndex]?.name;
              if (seriesName && seriesName !== 'Idle') {
                highlightJobSeriesAcrossCharts(seriesName);
              }
            },
            legendItemMouseOut: function(chartContext, seriesIndex, config) {
              resetJobSeriesHighlightAcrossCharts();
            },
            zoomed: function(chartContext, { xaxis }) {
              if (xaxis && xaxis.min && xaxis.max) {
                onTimelineDragZoom(xaxis.min, xaxis.max);
              }
            },
            selection: function(chartContext, { xaxis }) {
              if (xaxis && xaxis.min && xaxis.max) {
                onTimelineDragZoom(xaxis.min, xaxis.max);
              }
            }
          },
          animations: { enabled: false }
        },
        stroke: { curve: 'stepline', width: 2.5 },
        markers: { size: 0, hover: { size: 5 } },
        dataLabels: { enabled: false },
        series: series,
        colors: seriesColors,
        xaxis: {
          type: 'datetime',
          min: globalMinMs,
          max: globalMaxMs,
          labels: {
            datetimeUTC: false,
            style: { colors: '#64748b', fontSize: '11px' }
          }
        },
        yaxis: {
          min: 0,
          max: 1.1,
          tickAmount: 1,
          title: { text: 'GPU Activity' },
          labels: {
            minWidth: 55,
            maxWidth: 55,
            style: { colors: '#64748b', fontSize: '11px' },
            formatter: (val) => (val >= 0.9 ? 'Active' : 'Idle')
          }
        },
        legend: { show: true, position: 'bottom', horizontalAlign: 'center' },
        tooltip: {
          x: { format: 'HH:mm:ss.fff' }
        }
      };
      if (chartInstances[claimId]) {
        chartInstances[claimId].updateOptions(options, true, false);
      } else {
        chartInstances[claimId] = new ApexCharts(chartEl, options);
        chartInstances[claimId].render();
      }
    }

    let mutationChartInstance = null;

    function renderMutationChart(mutationHistory, weightSyncStrategy) {
      const cardEl = document.getElementById('mutation-chart-card');
      const chartEl = document.getElementById('mutation-chart-root');
      if (!cardEl || !chartEl) return;

      const isDelta = weightSyncStrategy === 'delta' || (mutationHistory && mutationHistory.length > 0);
      if (!isDelta || !mutationHistory || mutationHistory.length === 0) {
        cardEl.style.display = 'none';
        return;
      }

      cardEl.style.display = 'block';

      const steps = mutationHistory.map(m => `Step ${m.step}`);
      const values = mutationHistory.map(m => m.mutation_pct);

      const options = {
        series: [{
          name: '% Weight Mutation',
          data: values,
        }],
        chart: {
          type: 'area',
          height: 250,
          toolbar: { show: false },
          fontFamily: 'Inter, system-ui, sans-serif',
          sparkline: { enabled: false },
        },
        colors: ['#d97706'],
        stroke: {
          curve: 'smooth',
          width: 3,
        },
        fill: {
          type: 'gradient',
          gradient: {
            shadeIntensity: 1,
            opacityFrom: 0.35,
            opacityTo: 0.05,
            stops: [0, 95, 100],
          },
        },
        markers: {
          size: values.length <= 40 ? 5 : 3,
          colors: ['#d97706'],
          strokeColors: '#ffffff',
          strokeWidth: 2,
          hover: { size: 7 },
        },
        dataLabels: {
          enabled: values.length <= 20,
          formatter: (val) => `${val}%`,
          style: { fontSize: '10px', colors: ['#9a3412'], fontWeight: '600' },
          offsetY: -8,
        },
        xaxis: {
          categories: steps,
          labels: { style: { fontSize: '11px', colors: '#64748b' } },
          axisBorder: { show: true, color: '#e2e8f0' },
          axisTicks: { show: true, color: '#e2e8f0' },
        },
        yaxis: {
          title: { text: '% Mutated Params', style: { color: '#64748b', fontSize: '12px', fontWeight: 500 } },
          labels: {
            formatter: (val) => `${val.toFixed(2)}%`,
            style: { fontSize: '11px', colors: '#64748b' },
          },
        },
        grid: {
          borderColor: '#f1f5f9',
          strokeDashArray: 3,
          padding: {
            top: 15,
            right: 25,
            bottom: 10,
            left: 15,
          },
        },
        tooltip: {
          y: {
            formatter: (val, opts) => {
              const item = mutationHistory[opts.dataPointIndex];
              if (item && item.changed_elements !== undefined && item.total_elements !== undefined) {
                return `${val}% (${item.changed_elements.toLocaleString()} / ${item.total_elements.toLocaleString()} params)`;
              }
              return `${val}%`;
            },
          },
        },
      };

      if (mutationChartInstance) {
        mutationChartInstance.updateOptions(options, true, false);
      } else {
        mutationChartInstance = new ApexCharts(chartEl, options);
        mutationChartInstance.render();
      }
    }

    let selectedJobId = null;
    let activeJobFilter = 'all';
    let cachedJobsData = null;
    let cachedJobRequestsData = null;

    function filterJobs(filterType) {
      activeJobFilter = filterType;
      const btnAll = document.getElementById('btn-all-jobs');
      if (btnAll) btnAll.classList.toggle('active', filterType === 'all');
      document.getElementById('btn-active-jobs').classList.toggle('active', filterType === 'active');
      document.getElementById('btn-completed-jobs').classList.toggle('active', filterType === 'completed');
      if (cachedJobsData) renderJobsTable(cachedJobsData);
    }

    async function markJobCompleted(modelId) {
      try {
        await fetch('/api/v1/delete_model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_id: modelId })
        });
        fetchJobsList();
      } catch (e) {
        console.error('Failed to mark job completed', e);
      }
    }

    async function fetchJobsList() {
      try {
        const resp = await fetch('/api/v1/admin/jobs');
        if (!resp.ok) return;
        cachedJobsData = await resp.json();
        renderJobsTable(cachedJobsData);
        if (selectedJobId) {
          fetchJobRequests(selectedJobId);
        }
      } catch (e) {
        console.error('Failed to fetch jobs list', e);
      }
    }

    function formatTs(ts) {
      if (!ts || ts === 0) return '-';
      const d = new Date(ts * 1000);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    }

    function renderJobsTable(data) {
      const tbody = document.getElementById('jobs-table-body');
      if (!tbody) return;
      let rawJobs = [];
      if (activeJobFilter === 'active') {
        rawJobs = data.active_jobs || [];
      } else if (activeJobFilter === 'completed') {
        rawJobs = data.completed_jobs || [];
      } else {
        rawJobs = [...(data.active_jobs || []), ...(data.completed_jobs || [])];
      }
      const jobs = [...rawJobs].sort((a, b) => (b.created_at || 0) - (a.created_at || 0));

      if (jobs.length === 0) {
        const filterName = activeJobFilter === 'all' ? '' : `${activeJobFilter} `;
        tbody.innerHTML = `<tr><td colspan="9" style="padding: 24px; text-align: center; color: #94a3b8;">` +
          `No ${filterName}jobs found in cluster.</td></tr>`;
        return;
      }

      tbody.innerHTML = jobs.map(j => {
        const isAct = j.status === 'active';
        const activeSpan = `<span style="background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; ` +
          `padding: 3px 10px; border-radius: 9999px; font-weight: 600; font-size: 11px;">🟢 ACTIVE</span>`;
        const doneSpan = `<span style="background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; ` +
          `padding: 3px 10px; border-radius: 9999px; font-weight: 600; font-size: 11px;">⚪ COMPLETED</span>`;
        const statusBadge = isAct ? activeSpan : doneSpan;
        const queuesText = `${j.pending_trainer_reqs} Trainer / ${j.pending_sampler_reqs} Sampler`;
        const stepText = j.max_steps ? `Step ${j.current_step} / ${j.max_steps}` : `Step ${j.current_step}`;
        const isDelta = j.weight_sync_strategy === 'delta';
        const mutBadge = (isDelta && j.latest_mutation_pct !== undefined && j.latest_mutation_pct !== null)
          ? `<div style="font-size: 11px; color: #d97706; font-weight: 600;">⚡ Mutated ${j.latest_mutation_pct}% params</div>`
          : (isDelta ? `<div style="font-size: 11px; color: #d97706;">⚡ Delta Sync</div>` : '');

        return `
          <tr class="job-row"
              onmouseenter="highlightJobSeriesAcrossCharts(formatTenantId('${j.model_id}'))"
              onmouseleave="resetJobSeriesHighlightAcrossCharts()"
              onclick="inspectJob('${j.model_id}')"
              style="border-bottom: 1px solid #f1f5f9;">
            <td style="padding: 12px 10px; font-family: 'JetBrains Mono', monospace; ` +
              `font-weight: 700; color: #0284c7; font-size: 13px;">${j.model_id}</td>
            <td style="padding: 12px 10px; color: #1e293b; font-weight: 500;">${j.base_model}</td>
            <td style="padding: 12px 10px; color: #64748b;">${j.tenant_id}</td>
            <td style="padding: 12px 10px;">${statusBadge}</td>
            <td style="padding: 12px 10px; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #334155;">${stepText}${mutBadge}</td>
            <td style="padding: 12px 10px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #64748b;">${queuesText}</td>
            <td style="padding: 12px 10px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #64748b;">${formatTs(j.created_at)}</td>
            <td style="padding: 12px 10px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #64748b;">${formatTs(j.updated_at)}</td>
            <td style="padding: 12px 10px; display: flex; gap: 6px;">
              <button
                onclick="event.stopPropagation(); inspectJob('${j.model_id}')"
                style="background: #0284c7; color: white; border: none; border-radius: 6px; "
                style="padding: 5px 10px; font-size: 11px; font-weight: 600; cursor: pointer;"
              >Inspect →</button>
              ${isAct ? `<button onclick="event.stopPropagation(); markJobCompleted('${j.model_id}')" ` +
                `style="background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; border-radius: 6px; ` +
                `padding: 5px 10px; font-size: 11px; font-weight: 600; cursor: pointer;">Mark Completed</button>` : ''}
            </td>
          </tr>
        `;
      }).join('');
    }

    // Poll every 3s for smooth accel usage updates and 5s for jobs list
    fetchAndUpdate();
    fetchJobsList();
    setInterval(fetchAndUpdate, 3000);
    setInterval(fetchJobsList, 5000);
  </script>
</body>
</html>
"""
