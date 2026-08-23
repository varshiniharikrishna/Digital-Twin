/* ==========================================================================
   Dynamic Digital Twin — Frontend JavaScript (Full-Body Human Twin SVG)
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    let currentPatientId = null;
    let maxPatientWeek = 1;
    let clonesChartInstance = null;
    let currentBodyPart = "Knee";
    let currentInjuryType = "ACL Tear";
    const recalHistory = [];

    const patientSelect = document.getElementById('patientSelect');
    const weekSlider = document.getElementById('weekSlider');
    const weekSliderVal = document.getElementById('weekSliderVal');
    const downloadPdfBtn = document.getElementById('downloadPdfBtn');
    const recalForm = document.getElementById('recalForm');

    init();

    async function init() {
        setupTabs();
        await loadPatientList();
        setupEventListeners();
        await loadCurrentPatient();
        await loadEvaluationMetrics();
    }

    function setupTabs() {
        const tabs = document.querySelectorAll('.nav-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                tabs.forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                tab.classList.add('active');
                document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');

                if (tab.dataset.tab === 'treatmentClones') loadTreatmentClones();
                else if (tab.dataset.tab === 'twinOverview') loadTwinOverview();
                else if (tab.dataset.tab === 'recalibration') loadRecalibrationHistory();
            });
        });
    }

    function setupEventListeners() {
        patientSelect.addEventListener('change', async (e) => {
            currentPatientId = e.target.value;
            await loadCurrentPatient();
        });

        weekSlider.addEventListener('input', (e) => {
            const week = parseInt(e.target.value);
            weekSliderVal.textContent = `Week ${week}`;
            loadWeekDetail(week);
        });

        downloadPdfBtn.addEventListener('click', () => {
            window.location.href = `/api/patient/${currentPatientId}/pdf?week=${weekSlider.value}`;
        });

        recalForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await runRecalibration();
        });

        const toggleCheckinBtn = document.getElementById('toggleCheckinFormBtn');
        const checkinForm = document.getElementById('saveWeeklyCheckinForm');
        if (toggleCheckinBtn && checkinForm) {
            toggleCheckinBtn.addEventListener('click', () => {
                checkinForm.classList.toggle('hidden');
            });
        }

        if (checkinForm) {
            checkinForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const week = parseInt(weekSlider.value);
                const payload = {
                    week: week,
                    pain: parseFloat(document.getElementById('checkinPain').value),
                    rom: parseFloat(document.getElementById('checkinRom').value),
                    quad_strength: parseFloat(document.getElementById('checkinQuad').value),
                    ham_strength: parseFloat(document.getElementById('checkinHam').value),
                    balance: parseFloat(document.getElementById('checkinBalance').value),
                    walking_speed: parseFloat(document.getElementById('checkinSpeed').value),
                    swelling: document.getElementById('checkinSwelling').value,
                    compliance: parseFloat(document.getElementById('checkinCompliance').value),
                    patient_feedback: document.getElementById('checkinFeedback').value || "Observation recorded",
                    clinician_remark: document.getElementById('checkinRemark').value || "Saved to DB"
                };

                const statusMsg = document.getElementById('checkinStatusMsg');
                statusMsg.style.color = '#F59E0B';
                statusMsg.textContent = 'Saving to database...';

                try {
                    const res = await fetch(`/api/patient/${currentPatientId}/weekly/save`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const resData = await res.json();
                    if (res.ok && resData.success) {
                        statusMsg.style.color = '#10B981';
                        statusMsg.textContent = 'Observation saved to database and persisted.';
                        await loadWeekDetail(week);

                        // Show weekly recalibration result panel
                        const recal = resData.recalibration;
                        if (recal) {
                            renderWeeklyRecalPanel(recal);
                        }

                        setTimeout(() => { statusMsg.textContent = ''; }, 5000);
                    } else {
                        statusMsg.style.color = '#E11D48';
                        statusMsg.textContent = `Error: ${resData.error || 'Failed to save'}`;
                    }
                } catch (err) {
                    console.error(err);
                    statusMsg.style.color = '#E11D48';
                    statusMsg.textContent = 'Network error saving observation.';
                }
            });
        }
    }

    function renderWeeklyRecalPanel(recal) {
        const panel = document.getElementById('weeklyRecalPanel');
        panel.classList.remove('hidden');
        panel.scrollIntoView({ behavior: 'smooth', block: 'start' });

        document.getElementById('recalPanelWeek').textContent = recal.week;
        document.getElementById('recalPanelWeek2').textContent = recal.week;
        document.getElementById('recalPanelOldPct').textContent = `${recal.old_predicted_pct}%`;
        document.getElementById('recalPanelNewPct').textContent = `${recal.new_predicted_pct}%`;
        document.getElementById('recalPanelOldWks').textContent = `${recal.old_predicted_weeks} wks`;

        const newWksEl = document.getElementById('recalPanelNewWks');
        newWksEl.textContent = `${recal.new_predicted_weeks} wks (${recal.discharge_delta >= 0 ? '+' : ''}${recal.discharge_delta} wks)`;
        newWksEl.style.color = recal.discharge_delta <= 0 ? '#10B981' : '#E11D48';

        // Build Expected vs Real table
        const tbody = document.getElementById('evrWeeklyTbody');
        const metricLabels = {
            Pain: 'Pain Score (0-10)',
            ROM_deg: 'Range of Motion (deg)',
            Strength_pct: 'Quadriceps Strength (%)',
            Balance_Score: 'Balance Score',
            Walking_Speed_mps: 'Walking Speed (m/s)'
        };
        tbody.innerHTML = '';
        const evr = recal.expected_vs_real;
        Object.keys(evr).forEach(key => {
            const m = evr[key];
            const flagClass = m.flagged ? 'style="background: rgba(225,29,72,0.08);"' : '';
            const statusBadge = m.flagged
                ? `<span style="background:#E11D48; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.78rem;">FLAGGED</span>`
                : `<span style="background:#10B981; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.78rem;">On Track</span>`;
            const devColor = m.flagged ? '#E11D48' : '#10B981';
            tbody.innerHTML += `<tr ${flagClass}>
                <td><b>${metricLabels[key] || key}</b></td>
                <td>${m.expected}</td>
                <td><b>${m.actual}</b></td>
                <td style="color:${devColor}; font-weight:700;">${m.pct_deviation > 0 ? '+' : ''}${m.pct_deviation}%</td>
                <td>${statusBadge}</td>
            </tr>`;
        });

        document.getElementById('recalPanelExplanation').textContent = recal.explanation;

        const pathway = recal.recommended_pathway;
        document.getElementById('recalPanelPathway').textContent =
            pathway ? (pathway.Archetype_Label || pathway.Pathway_Name || 'Standard Protocol') : 'Standard Protocol';
    }

    async function loadRecalibrationHistory() {
        if (!currentPatientId) return;
        try {
            const res = await fetch(`/api/patient/${currentPatientId}/recalibration_history`);
            const data = await res.json();
            const tbody = document.getElementById('recalHistoryTbody');
            if (!tbody) return;
            if (!data.history || data.history.length === 0) {
                tbody.innerHTML = `<tr><td colspan="9" class="text-muted text-center">No weekly check-ins recorded yet for ${currentPatientId}.</td></tr>`;
                return;
            }
            tbody.innerHTML = '';
            data.history.forEach(h => {
                const delta = parseFloat(h.Recalibrated_Recovery_Weeks || 0) - parseFloat(h.Old_Predicted_Weeks || 0);
                const deltaStr = isNaN(delta) ? '-' : `${delta >= 0 ? '+' : ''}${delta.toFixed(1)} wks`;
                const deltaColor = delta <= 0 ? '#10B981' : '#E11D48';
                const notes = (h.Explanation || '').substring(0, 80) + ((h.Explanation || '').length > 80 ? '...' : '');
                tbody.innerHTML += `<tr>
                    <td><b>Week ${h.Week}</b></td>
                    <td>${h.Old_Predicted_Weeks ? h.Old_Predicted_Weeks + ' wks' : '-'}</td>
                    <td>${h.Recalibrated_Recovery_Weeks ? h.Recalibrated_Recovery_Weeks + ' wks' : '-'}</td>
                    <td style="color:${deltaColor}; font-weight:700;">${deltaStr}</td>
                    <td>${h.Pain_Expected != null ? h.Pain_Expected + ' / ' + h.Pain_Actual : '-'}</td>
                    <td>${h.ROM_Expected != null ? h.ROM_Expected + ' / ' + h.ROM_Actual : '-'}</td>
                    <td>${h.Strength_Expected != null ? h.Strength_Expected + ' / ' + h.Strength_Actual : '-'}</td>
                    <td style="color:#6366F1; font-weight:600;">${h.Updated_Pathway || 'Standard'}</td>
                    <td style="font-size:0.82rem; color:#64748B;">${notes}</td>
                </tr>`;
            });
        } catch (err) {
            console.error('Error loading recalibration history:', err);
        }
    }

    async function loadPatientList() {
        try {
            const res = await fetch('/api/patients');
            const data = await res.json();
            patientSelect.innerHTML = '';
            data.patients.forEach(pid => {
                const opt = document.createElement('option');
                opt.value = pid; opt.textContent = pid;
                patientSelect.appendChild(opt);
            });
            if (data.patients.length > 0) currentPatientId = data.patients[0];
        } catch (err) { console.error(err); }
    }

    async function loadCurrentPatient() {
        if (!currentPatientId) return;
        try {
            const resDetails = await fetch(`/api/patient/${currentPatientId}`);
            const dataDetails = await resDetails.json();
            currentBodyPart = dataDetails.patient.Body_Part || "Knee";
            currentInjuryType = dataDetails.patient.Injury_Type || "ACL Tear";

            const resWeekly = await fetch(`/api/patient/${currentPatientId}/weekly`);
            const weeklyData = await resWeekly.json();
            maxPatientWeek = weeklyData.max_week || 1;
            weekSlider.max = maxPatientWeek;
            weekSlider.value = 1;
            weekSliderVal.textContent = `Week 1`;
            await loadWeekDetail(1);
        } catch (err) { console.error(err); }
    }

    async function loadWeekDetail(week) {
        if (!currentPatientId) return;
        try {
            const res = await fetch(`/api/patient/${currentPatientId}/week/${week}`);
            if (!res.ok) return;
            const data = await res.json();
            const w = data.week_data;

            document.getElementById('mPain').textContent = `${w.Pain}/10`;
            document.getElementById('mRom').textContent = `${w.ROM_deg}°`;
            document.getElementById('mQuad').textContent = `${w.Quadriceps_Strength_pct}%`;
            document.getElementById('mHam').textContent = `${w.Hamstring_Strength_pct}%`;
            document.getElementById('mBalance').textContent = w.Balance_Score;
            document.getElementById('mSpeed').textContent = `${w.Walking_Speed_mps} m/s`;

            document.getElementById('mSwelling').textContent = w.Swelling;
            document.getElementById('mCompliance').textContent = `${w.Compliance_pct}%`;
            document.getElementById('mFeedback').textContent = w.Patient_Feedback;
            document.getElementById('mRemark').textContent = w.Clinician_Remark;

            renderFullBodyHumanSvg(parseFloat(w.Pain), currentBodyPart, data.muscle_compensation);
            renderPressurePoints(data.pressure_points);
            renderExpectedVsActual(data.expected_vs_actual);
            renderTherapyPlan(data.therapy_plan);
        } catch (err) { console.error(err); }
    }

    // ----------------------------------------------------------------------
    // FULL-BODY HUMAN ANATOMICAL SVG TWIN
    // ----------------------------------------------------------------------
    function renderFullBodyHumanSvg(painScore, bodyPart, compEvent) {
        const wrapper = document.getElementById('svgPainMapWrapper');
        
        let radius = 12;
        let fillColor = '#10B981';
        let glowColor = 'rgba(16, 185, 129, 0.3)';

        if (painScore >= 8.0) { radius = 26; fillColor = '#E11D48'; glowColor = 'rgba(225, 29, 72, 0.5)'; }
        else if (painScore >= 5.0) { radius = 20; fillColor = '#F59E0B'; glowColor = 'rgba(245, 158, 11, 0.4)'; }
        else if (painScore >= 2.0) { radius = 15; fillColor = '#EAB308'; glowColor = 'rgba(234, 179, 8, 0.3)'; }

        // Determine anatomical coordinates based on injured body part
        let spotX = 200, spotY = 295; // Default Knee
        let partLabel = "Right Knee";

        const bp = (bodyPart || "").toLowerCase();
        const inj = (currentInjuryType || "").toLowerCase();

        if (bp.includes("shoulder") || inj.includes("rotator")) { spotX = 250; spotY = 90; partLabel = "Right Shoulder"; }
        else if (bp.includes("ankle") || inj.includes("ankle") || inj.includes("achilles")) { spotX = 215; spotY = 405; partLabel = "Right Ankle"; }
        else if (bp.includes("lower leg") || inj.includes("tibial")) { spotX = 215; spotY = 350; partLabel = "Right Tibia / Calf"; }
        else if (bp.includes("arm") || bp.includes("elbow")) { spotX = 265; spotY = 135; partLabel = "Right Elbow / Arm"; }
        else if (bp.includes("wrist") || bp.includes("hand")) { spotX = 285; spotY = 185; partLabel = "Right Wrist"; }
        else if (bp.includes("neck") || bp.includes("head")) { spotX = 200; spotY = 55; partLabel = "Cervical Spine / Neck"; }
        else if (bp.includes("hip") || bp.includes("pelvis")) { spotX = 215; spotY = 205; partLabel = "Right Hip Joint"; }
        else { spotX = 215; spotY = 295; partLabel = "Right Knee Joint"; }

        let compMarker = '';
        if (compEvent) {
            compMarker = `
            <g transform="translate(270, ${spotY - 20})">
                <rect x="0" y="0" width="120" height="42" rx="6" fill="#FEF2F2" stroke="#EF4444" stroke-width="1.5"/>
                <text x="60" y="17" fill="#991B1B" font-size="10" font-weight="bold" text-anchor="middle">OVERLOAD ALERT</text>
                <text x="60" y="31" fill="#7F1D1D" font-size="9" text-anchor="middle">${compEvent.Muscle_Overloaded} (${compEvent.Compensation_Level})</text>
            </g>
            <line x1="${spotX}" y1="${spotY}" x2="270" y2="${spotY}" stroke="#EF4444" stroke-width="1.5" stroke-dasharray="3,3"/>
            `;
        }

        wrapper.innerHTML = `
        <svg width="400" height="460" viewBox="0 0 400 460">
            <defs>
                <filter id="glow"><feGaussianBlur stdDeviation="5" result="blur" /><feComposite in="SourceGraphic" in2="blur" operator="over" /></filter>
            </defs>

            <!-- Full-Body Human Anatomical Silhouette -->
            <!-- Head & Neck -->
            <ellipse cx="200" cy="35" rx="18" ry="22" fill="#E2E8F0" stroke="#94A3B8" stroke-width="1.5"/>
            <rect x="194" y="55" width="12" height="15" rx="2" fill="#CBD5E1"/>

            <!-- Torso / Chest / Spine / Pelvis -->
            <path d="M 165 70 L 235 70 L 225 180 L 175 180 Z" fill="#CBD5E1" stroke="#94A3B8" stroke-width="1.5"/>
            <line x1="200" y1="70" x2="200" y2="180" stroke="#94A3B8" stroke-width="1.5" stroke-dasharray="2,2"/>

            <!-- Left Arm -->
            <path d="M 165 70 L 140 85 L 125 135 L 115 185" fill="none" stroke="#CBD5E1" stroke-width="14" stroke-linecap="round"/>
            <circle cx="140" cy="85" r="8" fill="#94A3B8"/>
            <circle cx="125" cy="135" r="7" fill="#94A3B8"/>

            <!-- Right Arm -->
            <path d="M 235 70 L 260 85 L 275 135 L 285 185" fill="none" stroke="#CBD5E1" stroke-width="14" stroke-linecap="round"/>
            <circle cx="260" cy="85" r="8" fill="#94A3B8"/>
            <circle cx="275" cy="135" r="7" fill="#94A3B8"/>

            <!-- Hips / Pelvis -->
            <path d="M 175 180 L 225 180 L 230 210 L 170 210 Z" fill="#94A3B8"/>

            <!-- Left Leg -->
            <path d="M 180 210 L 180 290 L 180 380 L 180 430" fill="none" stroke="#CBD5E1" stroke-width="18" stroke-linecap="round"/>
            <circle cx="180" cy="295" r="9" fill="#94A3B8"/>
            <circle cx="180" cy="410" r="7" fill="#94A3B8"/>

            <!-- Right Leg -->
            <path d="M 220 210 L 220 290 L 220 380 L 220 430" fill="none" stroke="#CBD5E1" stroke-width="18" stroke-linecap="round"/>
            <circle cx="220" cy="295" r="9" fill="#94A3B8"/>
            <circle cx="220" cy="410" r="7" fill="#94A3B8"/>

            <!-- Dynamic Pain Spot & Pulsing Glow on Targeted Joint -->
            <circle cx="${spotX}" cy="${spotY}" r="${radius + 8}" fill="${glowColor}" filter="url(#glow)"/>
            <circle cx="${spotX}" cy="${spotY}" r="${radius}" fill="${fillColor}" stroke="#FFFFFF" stroke-width="2"/>
            <text x="${spotX}" y="${spotY + 4}" fill="#FFFFFF" font-size="10" font-weight="bold" text-anchor="middle">${painScore}</text>

            <!-- Anatomical Location Tag -->
            <rect x="15" y="15" width="130" height="26" rx="4" fill="#FFFFFF" stroke="#CBD5E1"/>
            <text x="80" y="32" fill="#0F172A" font-size="10" font-weight="bold" text-anchor="middle">${partLabel}</text>

            ${compMarker}
        </svg>
        `;
    }

    function renderPressurePoints(points) {
        const container = document.getElementById('pressureFlagsContainer');
        container.innerHTML = (!points || points.length === 0) ? `<div class="alert-box-info">✓ No critical pressure point flags detected!</div>` : '';
        if (points) points.forEach(p => {
            container.innerHTML += `<div class="flag-box"><div class="flag-title">ALERT: ${p.Category} — ${p.Flag}</div><div class="flag-detail">Recorded Value: <b>${p.Actual}</b></div></div>`;
        });
    }

    function renderExpectedVsActual(eva) {
        const container = document.getElementById('expectedVsActualContainer');
        if (!eva || Object.keys(eva).length === 0) return;
        let html = `<table class="data-table"><thead><tr><th>Metric</th><th>Expected</th><th>Actual</th><th>Deviation</th></tr></thead><tbody>`;
        Object.keys(eva).forEach(m => {
            const d = eva[m];
            html += `<tr><td><b>${m}</b></td><td>${d.predicted.toFixed(1)}</td><td>${d.actual.toFixed(1)}</td><td><strong style="color:${d.flagged ? '#E11D48' : '#059669'}">${d.pct_deviation >= 0 ? '+' : ''}${d.pct_deviation.toFixed(1)}%</strong></td></tr>`;
        });
        container.innerHTML = html + `</tbody></table>`;
    }

    function renderTherapyPlan(plan) {
        const container = document.getElementById('therapyPlanContainer');
        if (!plan || plan.length === 0) return;
        let html = `<table class="data-table"><thead><tr><th>Exercise</th><th>Target</th><th>Sets x Reps</th><th>Duration</th><th>Therapist Notes</th></tr></thead><tbody>`;
        plan.forEach(ex => {
            html += `<tr><td><b>${ex.Exercise_Name}</b></td><td>${ex.Target_Muscle}</td><td>${ex.Sets} &times; ${ex.Repetitions}</td><td>${ex.Duration_min} min</td><td class="italic-txt">${ex.Therapist_Notes || '-'}</td></tr>`;
        });
        container.innerHTML = html + `</tbody></table>`;
    }

    let treatmentTaxonomy = null;

    async function loadTreatmentClones() {
        if (!currentPatientId) return;

        if (!treatmentTaxonomy) {
            try {
                const resTax = await fetch('/api/treatment/taxonomy');
                treatmentTaxonomy = await resTax.json();

                const rxCategorySelect = document.getElementById('rxCategorySelect');
                const rxSubtypeSelect = document.getElementById('rxSubtypeSelect');
                const treatmentForm = document.getElementById('treatmentSimForm');

                if (rxCategorySelect) {
                    rxCategorySelect.addEventListener('change', (e) => {
                        updateSubtypeDropdown(e.target.value);
                    });
                    updateSubtypeDropdown(rxCategorySelect.value);
                }

                if (treatmentForm) {
                    treatmentForm.addEventListener('submit', async (e) => {
                        e.preventDefault();
                        await runTreatmentSimulation();
                    });
                }
            } catch (err) {
                console.error("Error loading taxonomy:", err);
            }
        }

        await runTreatmentSimulation();
    }

    function updateSubtypeDropdown(catKey) {
        const rxSubtypeSelect = document.getElementById('rxSubtypeSelect');
        if (!rxSubtypeSelect || !treatmentTaxonomy || !treatmentTaxonomy[catKey]) return;

        rxSubtypeSelect.innerHTML = '';
        const subMap = treatmentTaxonomy[catKey].subtypes;
        Object.keys(subMap).forEach(subKey => {
            const item = subMap[subKey];
            const opt = document.createElement('option');
            opt.value = subKey;
            opt.textContent = `${item.name} (${item.preferred_stage})`;
            rxSubtypeSelect.appendChild(opt);
        });
    }

    async function runTreatmentSimulation() {
        if (!currentPatientId) return;

        const category = document.getElementById('rxCategorySelect').value;
        const exercise_type = document.getElementById('rxSubtypeSelect').value;
        const stage = document.getElementById('rxStageSelect').value;
        const intensity = document.getElementById('rxIntensitySelect').value;

        try {
            const res = await fetch(`/api/patient/${currentPatientId}/treatment/simulate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ category, exercise_type, stage, intensity })
            });
            const data = await res.json();

            // Appropriateness Banner
            const banner = document.getElementById('rxApprBanner');
            banner.textContent = data.appropriateness.message;
            if (data.appropriateness.status === 'warning') {
                banner.className = 'flag-box mt-4';
            } else if (data.appropriateness.status === 'success') {
                banner.className = 'alert-box-info mt-4';
                banner.style.borderLeft = '4px solid #10B981';
            } else {
                banner.className = 'alert-box-info mt-4';
            }

            // Impact Cards
            document.getElementById('rxRomImpact').textContent = `+${data.impact.rom_boost.toFixed(1)}°`;
            document.getElementById('rxStrengthImpact').textContent = `+${data.impact.strength_boost.toFixed(1)}%`;
            document.getElementById('rxPainImpact').textContent = `${data.impact.pain_delta > 0 ? '+' : ''}${data.impact.pain_delta.toFixed(1)}`;
            
            const disEl = document.getElementById('rxDischargeImpact');
            disEl.textContent = `${data.impact.discharge_weeks_new} wks (${data.impact.discharge_delta >= 0 ? '+' : ''}${data.impact.discharge_delta} wks)`;
            disEl.style.color = data.impact.discharge_delta <= 0 ? '#10B981' : '#E11D48';

            // Summary text
            document.getElementById('rxSummaryText').textContent = data.reaction_summary;

            // Render Comparative Overlay Chart
            renderDivergentChart(data.baseline_trajectory, data.treatment_trajectory, data.exercise.name);

        } catch (err) {
            console.error("Error simulating treatment:", err);
        }
    }

    function renderDivergentChart(baseTraj, txTraj, exerciseName) {
        const ctx = document.getElementById('clonesChart').getContext('2d');
        if (clonesChartInstance) clonesChartInstance.destroy();

        const maxWeeks = Math.max(baseTraj.length, txTraj.length);
        const labels = Array.from({length: maxWeeks}, (_, i) => `Week ${i+1}`);

        clonesChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Standard Recovery (Strength %)',
                        data: baseTraj.map(t => t.Predicted_Strength_pct),
                        borderColor: '#64748B',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        fill: false
                    },
                    {
                        label: `Treatment Stimulated: ${exerciseName} (Strength %)`,
                        data: txTraj.map(t => t.Predicted_Strength_pct),
                        borderColor: '#10B981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 3,
                        tension: 0.3
                    },
                    {
                        label: `Treatment Pain Curve (Pain Score 0-10)`,
                        data: txTraj.map(t => t.Predicted_Pain),
                        borderColor: '#E11D48',
                        borderWidth: 2,
                        tension: 0.3,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#0F172A', font: { weight: 'bold' } } }
                },
                scales: {
                    x: { ticks: { color: '#64748B' }, grid: { color: '#E2E8F0' } },
                    y: {
                        title: { display: true, text: 'Strength (%)', color: '#0F172A' },
                        ticks: { color: '#64748B' }, grid: { color: '#E2E8F0' }, min: 0, max: 100
                    },
                    y1: {
                        position: 'right',
                        title: { display: true, text: 'Pain (0-10)', color: '#E11D48' },
                        ticks: { color: '#E11D48' }, grid: { drawOnChartArea: false }, min: 0, max: 10
                    }
                }
            }
        });
    }

    async function loadTwinOverview() {
        if (!currentPatientId) return;
        const res = await fetch(`/api/patient/${currentPatientId}`);
        const data = await res.json();
        const p = data.patient, pred = data.prediction;

        document.getElementById('realPatientInfoTable').innerHTML = `<tr><td>Patient ID</td><td>${p.Patient_ID}</td></tr><tr><td>Age / Gender</td><td>${p.Age} / ${p.Gender}</td></tr><tr><td>Injury Type</td><td>${p.Injury_Type}</td></tr><tr><td>Body Part</td><td>${p.Body_Part}</td></tr><tr><td>Severity</td><td>${p.Severity}</td></tr><tr><td>Baseline Pain / ROM</td><td>${p.Pain_Score}/10 | ${p.Range_of_Motion_deg}°</td></tr>`;
        document.getElementById('digitalTwinStateTable').innerHTML = `<tr><td>Twin ID</td><td>DT_${p.Patient_ID}</td></tr><tr><td>Predicted Recovery %</td><td class="text-emerald"><b>${pred.predicted_pct}%</b></td></tr><tr><td>Predicted Duration</td><td class="text-emerald"><b>${pred.predicted_weeks} wks</b></td></tr><tr><td>Model R² Confidence</td><td>${pred.r2_score}</td></tr>`;

        let simHtml = `<table class="data-table"><thead><tr><th>Patient ID</th><th>Similarity</th><th>Age / Gender</th><th>Injury</th><th>Severity</th><th>Recovery %</th><th>Duration</th></tr></thead><tbody>`;
        data.similar_patients.forEach(sp => {
            simHtml += `<tr><td><b>${sp.Patient_ID}</b></td><td class="text-emerald"><b>${(sp.Similarity_Score * 100).toFixed(0)}%</b></td><td>${sp.Age} / ${sp.Gender}</td><td>${sp.Injury_Type}</td><td>${sp.Severity}</td><td>${sp.Final_Recovery_Pct.toFixed(0)}%</td><td>${sp.Actual_Recovery_Time_weeks.toFixed(0)} wks</td></tr>`;
        });
        document.getElementById('similarPatientsTableWrapper').innerHTML = simHtml + `</tbody></table>`;
    }

    async function runRecalibration() {
        const week = document.getElementById('recalWeek').value;
        const pain = document.getElementById('recalPain').value;
        const rom = document.getElementById('recalRom').value;
        const strength = document.getElementById('recalStrength').value;

        const res = await fetch(`/api/patient/${currentPatientId}/recalibrate`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ week, pain, rom, strength })
        });
        const data = await res.json();

        // Also persist recalibration to database
        fetch(`/api/patient/${currentPatientId}/recalibrate/save`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ week, pain, rom, strength })
        }).catch(err => console.error("Error persisting recalibration:", err));

        document.getElementById('oldPredPct').textContent = `${data.old_predicted_pct}%`;
        document.getElementById('newPredPct').textContent = `${data.new_predicted_pct}%`;
        document.getElementById('oldPredWks').textContent = `${data.old_predicted_weeks} wks`;
        document.getElementById('newPredWks').textContent = `${data.new_predicted_weeks} wks`;
        document.getElementById('recalExplanation').textContent = data.explanation;
        document.getElementById('recalResultCard').classList.remove('hidden');

        recalHistory.push({ week, oldPred: `${data.old_predicted_pct}%`, newPred: `${data.new_predicted_pct}%`, pathway: data.recommended_pathway ? data.recommended_pathway.Archetype_Label : 'Standard' });
        renderRecalHistory();
    }

    function renderRecalHistory() {
        const tbody = document.getElementById('recalHistoryTbody');
        tbody.innerHTML = '';
        recalHistory.forEach(item => {
            tbody.innerHTML += `<tr><td>Week ${item.week}</td><td>${item.oldPred}</td><td class="text-emerald"><b>${item.newPred}</b></td><td>${item.pathway}</td></tr>`;
        });
    }

    async function loadEvaluationMetrics() {
        const res = await fetch('/api/metrics');
        const data = await res.json();

        document.getElementById('kpiPctR2').textContent = data.metrics_pct.R2.toFixed(3);
        document.getElementById('kpiPctRmse').textContent = data.metrics_pct.RMSE.toFixed(2);
        document.getElementById('kpiWksR2').textContent = data.metrics_weeks.R2.toFixed(3);
        document.getElementById('kpiWksRmse').textContent = data.metrics_weeks.RMSE.toFixed(2);

        const fiContainer = document.getElementById('featureImportanceList');
        fiContainer.innerHTML = '';
        const maxImp = Math.max(...data.feature_importance.map(f => f.Importance));
        data.feature_importance.forEach(f => {
            fiContainer.innerHTML += `<div class="fi-item"><div class="fi-name">${f.Feature}</div><div class="fi-bar-wrapper"><div class="fi-bar-fill" style="width: ${(f.Importance / maxImp * 100).toFixed(0)}%"></div></div><div style="font-size:0.85rem; font-weight:600; width:45px;">${(f.Importance * 100).toFixed(1)}%</div></div>`;
        });
    }
});
