/* ==========================================================================
   Dynamic Digital Twin — Frontend JavaScript
   Features Multi-Portal RBAC Authentication, Scope Authorization Headers,
   Doctor Roster Management (Admin only), Allocated Patient Filtering,
   Pathway Mining (Clinicians/Admins only), Treatment Simulator, and Recalibration.
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    let currentAuthToken = sessionStorage.getItem('dt_token') || null;
    let currentUserRole = sessionStorage.getItem('dt_role') || null;
    let currentUserId = sessionStorage.getItem('dt_user_id') || null;
    let currentLinkedDoc = sessionStorage.getItem('dt_linked_doc') || null;
    let currentLinkedPat = sessionStorage.getItem('dt_linked_pat') || null;

    let currentPatientId = null;
    let maxPatientWeek = 1;
    let clonesChartInstance = null;
    let currentBodyPart = "Knee";
    let currentInjuryType = "ACL Tear";

    const loginOverlay = document.getElementById('loginOverlay');
    const securityBanner = document.getElementById('securityBanner');
    const sessionUserName = document.getElementById('sessionUserName');
    const sessionRoleBadge = document.getElementById('sessionRoleBadge');
    const sessionScopeDesc = document.getElementById('sessionScopeDesc');
    const logoutBtn = document.getElementById('logoutBtn');
    const changePasswordBtn = document.getElementById('changePasswordBtn');

    const patientSelect = document.getElementById('patientSelect');
    const weekSlider = document.getElementById('weekSlider');
    const weekSliderVal = document.getElementById('weekSliderVal');
    const downloadPdfBtn = document.getElementById('downloadPdfBtn');
    const recalForm = document.getElementById('recalForm');

    init();

    async function init() {
        setupAuthForms();
        setupTabs();
        setupEventListeners();

        if (currentAuthToken) {
            hideLoginOverlay();
            await applyRolePermissions();
        } else {
            showLoginOverlay();
        }
    }

    // ----------------------------------------------------------------------
    // AUTHENTICATION & REST FETCH HELPERS
    // ----------------------------------------------------------------------
    async function fetchWithAuth(url, options = {}) {
        options.headers = options.headers || {};
        if (currentAuthToken) {
            options.headers['Authorization'] = `Bearer ${currentAuthToken}`;
        }
        const res = await fetch(url, options);
        if (res.status === 401) {
            logout();
            throw new Error("Session expired. Please log in again.");
        }
        return res;
    }

    function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }

    function showLoginOverlay() {
        if (loginOverlay) loginOverlay.classList.remove('hidden');
        if (securityBanner) securityBanner.classList.add('hidden');
    }

    function hideLoginOverlay() {
        if (loginOverlay) loginOverlay.classList.add('hidden');
        if (securityBanner) securityBanner.classList.remove('hidden');
    }

    function setupAuthForms() {
        const patientForm = document.getElementById('patientLoginForm');
        if (patientForm) {
            patientForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const inp = document.getElementById('patientLoginId').value;
                const pwd = document.getElementById('patientPassword').value;
                await performLogin(inp, pwd, 'PATIENT');
            });
        }

        const docForm = document.getElementById('doctorLoginForm');
        if (docForm) {
            docForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const inp = document.getElementById('docLoginId').value;
                const pwd = document.getElementById('docPassword').value;
                await performLogin(inp, pwd, 'DOCTOR');
            });
        }

        const adminForm = document.getElementById('adminLoginForm');
        if (adminForm) {
            adminForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const inp = document.getElementById('adminLoginId').value;
                const pwd = document.getElementById('adminPassword').value;
                await performLogin(inp, pwd, 'ADMIN');
            });
        }

        if (logoutBtn) {
            logoutBtn.addEventListener('click', logout);
        }
        if (changePasswordBtn) {
            changePasswordBtn.addEventListener('click', async () => {
                const current = window.prompt('Current password:');
                if (current === null) return;
                const next = window.prompt('New password (at least 12 characters):');
                if (next === null) return;
                const res = await fetchWithAuth('/api/auth/password', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ current_password: current, new_password: next })
                });
                const result = await res.json();
                window.alert(res.ok ? result.message : (result.error || 'Password update failed.'));
            });
        }
    }

    async function performLogin(loginInput, password, expectedRole) {
        const statusMsg = document.getElementById('loginStatusMsg');
        if (statusMsg) {
            statusMsg.style.color = '#F59E0B';
            statusMsg.textContent = 'Authenticating with Digital Twin database...';
        }

        try {
            const res = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ login_input: loginInput, password: password, expected_role: expectedRole })
            });
            const data = await res.json();

            if (res.ok && data.status === 'success') {
                currentAuthToken = data.token;
                currentUserRole = data.role;
                currentUserId = data.user_id;
                currentLinkedDoc = data.linked_doctor_id;
                currentLinkedPat = data.linked_patient_id;

                sessionStorage.setItem('dt_token', currentAuthToken);
                sessionStorage.setItem('dt_role', currentUserRole);
                sessionStorage.setItem('dt_user_id', currentUserId);
                sessionStorage.setItem('dt_linked_doc', currentLinkedDoc || '');
                sessionStorage.setItem('dt_linked_pat', currentLinkedPat || '');

                updateSecurityBanner(data);
                hideLoginOverlay();
                await applyRolePermissions();
            } else {
                if (statusMsg) {
                    statusMsg.style.color = '#E11D48';
                    statusMsg.textContent = `Login Failed: ${data.error || 'Invalid credentials'}`;
                }
            }
        } catch (err) {
            console.error(err);
            if (statusMsg) {
                statusMsg.style.color = '#E11D48';
                statusMsg.textContent = 'Network or server error during login.';
            }
        }
    }

    function updateSecurityBanner(authData) {
        if (!securityBanner) return;
        securityBanner.classList.remove('hidden');

        if (sessionUserName) sessionUserName.textContent = authData.username || currentUserId;
        if (sessionRoleBadge) {
            sessionRoleBadge.textContent = authData.role || currentUserRole;
            sessionRoleBadge.className = `session-badge ${currentUserRole === 'ADMIN' ? 'badge-amber' : (currentUserRole === 'DOCTOR' ? 'badge-emerald' : 'badge-purple')}`;
        }
        if (sessionScopeDesc) sessionScopeDesc.textContent = authData.security_scope || '';

        const admBtn = document.getElementById('adminQuickAddDocBtn');
        const docBtn = document.getElementById('docQuickAddPatBtn');

        // Only Admin can see + Add Doctor button
        if (admBtn) admBtn.classList.toggle('hidden', currentUserRole !== 'ADMIN');
        if (docBtn) docBtn.classList.toggle('hidden', !['ADMIN', 'DOCTOR'].includes(currentUserRole));
    }

    function logout() {
        if (currentAuthToken) {
            fetch('/api/auth/logout', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${currentAuthToken}` }
            }).catch(err => console.error(err));
        }

        sessionStorage.clear();
        currentAuthToken = null;
        currentUserRole = null;
        currentUserId = null;
        currentLinkedDoc = null;
        currentLinkedPat = null;

        showLoginOverlay();
    }

    // ----------------------------------------------------------------------
    // ROLE PORTALS & ACCESS ENFORCEMENT
    // ----------------------------------------------------------------------
    async function applyRolePermissions() {
        updateSecurityBanner({
            username: currentUserId,
            role: currentUserRole,
            security_scope: currentUserRole === 'ADMIN' ? 'Full System-Wide Access'
                          : currentUserRole === 'DOCTOR' ? `Restricted to Assigned Patients (${currentLinkedDoc})`
                          : `Personal Scope (${currentLinkedPat})`
        });

        // Tab Navigation Access Controls:
        const pathwaysTabBtn = document.getElementById('tabNavPathways');
        const docTabBtn = document.getElementById('tabNavDoctorRoster');
        const admTabBtn = document.getElementById('tabNavAdminPortal');

        // Patients CANNOT see Pathway Mining tab
        if (pathwaysTabBtn) pathwaysTabBtn.classList.toggle('hidden', currentUserRole === 'PATIENT');
        
        // Only Admin can see Doctor Roster tab and Admin Control Center
        if (docTabBtn) docTabBtn.classList.toggle('hidden', currentUserRole !== 'ADMIN');
        if (admTabBtn) admTabBtn.classList.toggle('hidden', currentUserRole !== 'ADMIN');

        await loadPatientList();

        if (['ADMIN', 'DOCTOR'].includes(currentUserRole)) {
            await loadDoctorRoster();
        }
        if (currentUserRole === 'ADMIN') {
            await loadAdminOverview();
        }

        await loadCurrentPatient();
        await loadEvaluationMetrics();
    }

    async function loadPatientList() {
        try {
            const res = await fetchWithAuth('/api/patients');
            const data = await res.json();
            if (patientSelect) {
                patientSelect.innerHTML = '';
                (data.patients || []).forEach(pid => {
                    const opt = document.createElement('option');
                    opt.value = pid;
                    opt.textContent = pid;
                    patientSelect.appendChild(opt);
                });
            }
            if (data.patients && data.patients.length > 0) {
                currentPatientId = data.patients[0];
            }
        } catch (err) { console.error(err); }
    }

    async function loadCurrentPatient() {
        if (!currentPatientId) return;
        try {
            const resDetails = await fetchWithAuth(`/api/patient/${currentPatientId}`);
            if (!resDetails.ok) {
                console.warn("Could not load patient details:", resDetails.status);
                return;
            }
            const dataDetails = await resDetails.json();
            
            currentBodyPart = dataDetails.patient.Body_Part || "Knee";
            currentInjuryType = dataDetails.patient.Injury_Type || "ACL Tear";

            const resWeekly = await fetchWithAuth(`/api/patient/${currentPatientId}/weekly`);
            const weeklyData = await resWeekly.json();
            maxPatientWeek = weeklyData.max_week || 1;
            
            if (weekSlider) {
                weekSlider.max = maxPatientWeek;
                weekSlider.value = 1;
            }
            if (weekSliderVal) weekSliderVal.textContent = `Week 1`;
            
            await loadWeekDetail(1);
        } catch (err) { console.error(err); }
    }

    async function loadWeekDetail(week) {
        if (!currentPatientId) return;
        try {
            const res = await fetchWithAuth(`/api/patient/${currentPatientId}/week/${week}`);
            if (!res.ok) return;
            const data = await res.json();
            const w = data.week_data || {};

            const painVal = (w.Pain !== undefined && w.Pain !== null) ? parseFloat(w.Pain).toFixed(1) : "0.0";
            const romVal = (w.ROM_deg !== undefined && w.ROM_deg !== null) ? parseFloat(w.ROM_deg).toFixed(1) : "0.0";
            const quadVal = (w.Quadriceps_Strength_pct !== undefined && w.Quadriceps_Strength_pct !== null) ? parseFloat(w.Quadriceps_Strength_pct).toFixed(1) : "0.0";
            const hamVal = (w.Hamstring_Strength_pct !== undefined && w.Hamstring_Strength_pct !== null) ? parseFloat(w.Hamstring_Strength_pct).toFixed(1) : "0.0";
            const balVal = (w.Balance_Score !== undefined && w.Balance_Score !== null) ? parseFloat(w.Balance_Score).toFixed(1) : "0.0";
            const speedVal = (w.Walking_Speed_mps !== undefined && w.Walking_Speed_mps !== null) ? parseFloat(w.Walking_Speed_mps).toFixed(2) : "0.00";

            if (document.getElementById('mPain')) document.getElementById('mPain').textContent = `${painVal}/10`;
            if (document.getElementById('mRom')) document.getElementById('mRom').textContent = `${romVal}°`;
            if (document.getElementById('mQuad')) document.getElementById('mQuad').textContent = `${quadVal}%`;
            if (document.getElementById('mHam')) document.getElementById('mHam').textContent = `${hamVal}%`;
            if (document.getElementById('mBalance')) document.getElementById('mBalance').textContent = balVal;
            if (document.getElementById('mSpeed')) document.getElementById('mSpeed').textContent = `${speedVal} m/s`;

            if (document.getElementById('mSwelling')) document.getElementById('mSwelling').textContent = w.Swelling || "None";
            if (document.getElementById('mCompliance')) document.getElementById('mCompliance').textContent = `${w.Compliance_pct || 100}%`;
            if (document.getElementById('mFeedback')) document.getElementById('mFeedback').textContent = w.Patient_Feedback || "On track.";
            if (document.getElementById('mRemark')) document.getElementById('mRemark').textContent = w.Clinician_Remark || "Progressing normally.";

            if (data.npi && document.getElementById('npiBadgeScore')) {
                document.getElementById('npiBadgeScore').textContent = data.npi.npi_score.toFixed(3);
                document.getElementById('npiRawPain').textContent = `${data.npi.raw_pain}/10`;
                document.getElementById('npiScoreVal').textContent = data.npi.npi_score.toFixed(3);
                document.getElementById('npiCategory').textContent = data.npi.category;
                document.getElementById('npiPctVal').textContent = `${data.npi.npi_pct}%`;
            }

            renderFullBodyHumanSvg(parseFloat(painVal), currentBodyPart, data.muscle_compensation, romVal, quadVal);
            renderPressurePoints(data.pressure_points);
            renderExpectedVsActual(data.expected_vs_actual);
            renderTherapyPlan(data.therapy_plan);
        } catch (err) { console.error(err); }
    }

    // ----------------------------------------------------------------------
    // TAB MANAGEMENT & EVENT LISTENERS
    // ----------------------------------------------------------------------
    function setupTabs() {
        const tabs = document.querySelectorAll('.nav-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                tabs.forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                tab.classList.add('active');
                const targetTab = document.getElementById(`tab-${tab.dataset.tab}`);
                if (targetTab) targetTab.classList.add('active');

                if (tab.dataset.tab === 'treatmentClones') loadTreatmentClones();
                else if (tab.dataset.tab === 'pathways') loadMinedPathways();
                else if (tab.dataset.tab === 'twinOverview') loadTwinOverview();
                else if (tab.dataset.tab === 'recalibration') loadRecalibrationHistory();
                else if (tab.dataset.tab === 'doctorRoster') loadDoctorRoster();
                else if (tab.dataset.tab === 'adminPortal') loadAdminOverview();
            });
        });
    }

    function setupEventListeners() {
        const explainBtn = document.getElementById('explainPatientBtn');
        if (explainBtn) {
            explainBtn.addEventListener('click', async () => {
                const target = document.getElementById('patientExplanation');
                if (!currentPatientId) { target.textContent = 'Select a patient first.'; return; }
                target.textContent = 'Calculating SHAP values…';
                try {
                    const response = await fetchWithAuth(`/api/patient/${encodeURIComponent(currentPatientId)}/explanation`);
                    const result = await response.json();
                    if (!response.ok) throw new Error(result.error || 'Explanation unavailable.');
                    const rows = result.features.map(item => `<tr><td>${item.feature}</td><td>${Number(item.shap_value).toFixed(4)}</td></tr>`).join('');
                    target.innerHTML = `<p>Predicted recovery: ${Number(result.prediction).toFixed(1)}%. Base value: ${Number(result.base_value).toFixed(1)}%.</p><p>${result.caveat}</p><div class="table-wrapper"><table class="data-table"><thead><tr><th>Model feature</th><th>SHAP value</th></tr></thead><tbody>${rows}</tbody></table></div>`;
                } catch (err) { target.textContent = err.message; }
            });
        }

        if (patientSelect) {
            patientSelect.addEventListener('change', async (e) => {
                currentPatientId = e.target.value;
                await loadCurrentPatient();
            });
        }

        if (weekSlider) {
            weekSlider.addEventListener('input', (e) => {
                const week = parseInt(e.target.value);
                weekSliderVal.textContent = `Week ${week}`;
                loadWeekDetail(week);
            });
        }

        if (downloadPdfBtn) {
            downloadPdfBtn.addEventListener('click', async () => {
                const res = await fetchWithAuth(`/api/patient/${currentPatientId}/pdf?week=${weekSlider.value}`);
                if (!res.ok) throw new Error('Could not download the patient document.');
                const objectUrl = URL.createObjectURL(await res.blob());
                const link = document.createElement('a');
                link.href = objectUrl;
                link.download = `Digital_Twin_Report_${currentPatientId}_W${weekSlider.value}.pdf`;
                link.click();
                URL.revokeObjectURL(objectUrl);
            });
        }

        // Modals
        const addDocModal = document.getElementById('addDoctorModal');
        const openAddDocBtn = document.getElementById('addDoctorModalOpenBtn');
        const quickAddDocBtn = document.getElementById('adminQuickAddDocBtn');
        const closeAddDocBtn = document.getElementById('closeAddDocModalBtn');
        const cancelAddDocBtn = document.getElementById('cancelAddDocModalBtn');

        [openAddDocBtn, quickAddDocBtn].forEach(btn => {
            if (btn) btn.addEventListener('click', () => addDocModal.classList.remove('hidden'));
        });
        [closeAddDocBtn, cancelAddDocBtn].forEach(btn => {
            if (btn) btn.addEventListener('click', () => addDocModal.classList.add('hidden'));
        });

        const addDoctorForm = document.getElementById('addDoctorForm');
        if (addDoctorForm) {
            addDoctorForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const payload = {
                    Doctor_Name: document.getElementById('newDocName').value,
                    Specialization: document.getElementById('newDocSpec').value,
                    Hospital_Branch: document.getElementById('newDocBranch').value,
                    Years_Experience: parseInt(document.getElementById('newDocExp').value),
                    Contact_Email: document.getElementById('newDocEmail').value
                };
                try {
                    const res = await fetchWithAuth('/api/admin/doctor/add', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();
                    if (res.ok) {
                        alert(`${data.message}\nLogin ID: ${data.doctor_id}\nTemporary password: ${data.initial_password}\nShare this once and ask the doctor to change it after signing in.`);
                        addDocModal.classList.add('hidden');
                        await loadDoctorRoster();
                    } else alert(`Error: ${data.error}`);
                } catch (err) { alert(`Failed: ${err}`); }
            });
        }

        // Register Patient Modal (DOCTOR & ADMIN)
        const regPatModal = document.getElementById('registerPatientModal');
        const docQuickAddPatBtn = document.getElementById('docQuickAddPatBtn');
        const closeRegPatModalBtn = document.getElementById('closeRegPatModalBtn');
        const cancelRegPatModalBtn = document.getElementById('cancelRegPatModalBtn');

        if (docQuickAddPatBtn) {
            docQuickAddPatBtn.addEventListener('click', async () => {
                await loadDoctorRoster();
                if (regPatModal) regPatModal.classList.remove('hidden');
            });
        }
        [closeRegPatModalBtn, cancelRegPatModalBtn].forEach(btn => {
            if (btn) btn.addEventListener('click', () => {
                if (regPatModal) regPatModal.classList.add('hidden');
            });
        });

        const registerPatientForm = document.getElementById('registerPatientForm');
        if (registerPatientForm) {
            registerPatientForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const payload = {
                    Age: parseInt(document.getElementById('regAge').value),
                    Gender: document.getElementById('regGender').value,
                    Height_cm: parseFloat(document.getElementById('regHeight').value),
                    Weight_kg: parseFloat(document.getElementById('regWeight').value),
                    Occupation: document.getElementById('regOccupation').value,
                    Activity_Level: document.getElementById('regActivity').value,
                    Body_Region: document.getElementById('regBodyRegion').value,
                    Injury_Type: document.getElementById('regInjuryType').value,
                    Severity: document.getElementById('regSeverity').value,
                    Surgery_Required: parseInt(document.getElementById('regSurgery').value),
                    Pain_Score: parseFloat(document.getElementById('regPain').value),
                    Primary_ROM_deg: parseFloat(document.getElementById('regRom').value),
                    Assigned_Doctor_ID: document.getElementById('regAssignedDocSelect').value || currentLinkedDoc || 'DOC0001'
                };
                try {
                    const res = await fetchWithAuth('/api/doctor/patient/add', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();
                    if (res.ok) {
                        alert(`${data.message}\nLogin ID: ${data.patient_id}\nTemporary password: ${data.initial_password}\nShare this securely and ask the patient to change it after signing in.`);
                        if (regPatModal) regPatModal.classList.add('hidden');
                        await loadPatientList();
                        if (data.patient_id) {
                            const patSelect = document.getElementById('patientSelect');
                            if (patSelect) {
                                patSelect.value = data.patient_id;
                                patSelect.dispatchEvent(new Event('change'));
                            }
                        }
                    } else alert(`Error: ${data.error}`);
                } catch (err) { alert(`Registration failed: ${err}`); }
            });
        }

        const adminRegisterPatientForm = document.getElementById('adminRegisterPatientForm');
        if (adminRegisterPatientForm) {
            adminRegisterPatientForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const payload = {
                    Age: parseInt(document.getElementById('admAge').value),
                    Gender: document.getElementById('admGender').value,
                    Body_Region: document.getElementById('admBodyRegion').value,
                    Injury_Type: document.getElementById('admInjuryType').value,
                    Severity: document.getElementById('admSeverity').value,
                    Pain_Score: parseFloat(document.getElementById('admPain').value),
                    Assigned_Doctor_ID: document.getElementById('admAssignedDocSelect').value
                };
                try {
                    const res = await fetchWithAuth('/api/doctor/patient/add', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();
                    if (res.ok) {
                        alert(`${data.message}\nLogin ID: ${data.patient_id}\nTemporary password: ${data.initial_password}\nShare this securely and ask the patient to change it after signing in.`);
                        await loadPatientList();
                        if (currentUserRole === 'ADMIN') await loadAdminOverview();
                    } else alert(`Error: ${data.error}`);
                } catch (err) { alert(`Registration failed: ${err}`); }
            });
        }

        const adminReassignForm = document.getElementById('adminReassignForm');
        if (adminReassignForm) {
            adminReassignForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const pid = document.getElementById('reassignPatientSelect').value;
                const docid = document.getElementById('reassignDoctorSelect').value;
                try {
                    const res = await fetchWithAuth('/api/admin/patient/reassign', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ patient_id: pid, new_doctor_id: docid })
                    });
                    const data = await res.json();
                    if (res.ok) alert(data.message);
                    else alert(`Error: ${data.error}`);
                } catch (err) { alert(`Failed: ${err}`); }
            });
        }

        const adminResetPasswordForm = document.getElementById('adminResetPasswordForm');
        if (adminResetPasswordForm) {
            adminResetPasswordForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const res = await fetchWithAuth('/api/admin/account/reset-password', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ login_input: document.getElementById('resetAccountId').value })
                });
                const data = await res.json();
                window.alert(res.ok
                    ? `Login ID: ${data.login_id}\nOne-time password: ${data.initial_password}\n${data.message}`
                    : (data.error || 'Password reset failed.'));
                if (res.ok) adminResetPasswordForm.reset();
            });
        }

        const checkinForm = document.getElementById('saveWeeklyCheckinForm');
        const toggleCheckinBtn = document.getElementById('toggleCheckinFormBtn');
        if (toggleCheckinBtn && checkinForm) {
            toggleCheckinBtn.addEventListener('click', () => checkinForm.classList.toggle('hidden'));
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
                if (statusMsg) {
                    statusMsg.style.color = '#F59E0B';
                    statusMsg.textContent = 'Saving observation to MySQL database...';
                }

                try {
                    const res = await fetchWithAuth(`/api/patient/${currentPatientId}/weekly/save`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const resData = await res.json();
                    if (res.ok && resData.success) {
                        if (statusMsg) {
                            statusMsg.style.color = '#10B981';
                            statusMsg.textContent = 'Observation saved and twin recalibrated!';
                        }
                        await loadWeekDetail(week);
                        if (resData.recalibration) renderWeeklyRecalPanel(resData.recalibration);
                        setTimeout(() => { if (statusMsg) statusMsg.textContent = ''; }, 5000);
                    } else {
                        if (statusMsg) {
                            statusMsg.style.color = '#E11D48';
                            statusMsg.textContent = `Error: ${resData.error || 'Save failed'}`;
                        }
                    }
                } catch (err) {
                    if (statusMsg) {
                        statusMsg.style.color = '#E11D48';
                        statusMsg.textContent = 'Network error saving observation.';
                    }
                }
            });
        }
    }

    // ----------------------------------------------------------------------
    // DYNAMIC PATHWAY MINING RENDERER (CLINICIANS / ADMINS)
    // ----------------------------------------------------------------------
    async function loadMinedPathways() {
        const wrapper = document.getElementById('minedPathwaysCardsWrapper');
        if (!wrapper) return;
        try {
            const res = await fetchWithAuth('/api/pathways');
            if (res.status === 403) {
                wrapper.innerHTML = `<div class="alert-box-info" style="border-left-color:#E11D48; color:#991B1B;">🔒 Access Restricted: Pathway Mining is restricted to clinicians and administrators.</div>`;
                return;
            }
            const data = await res.json();
            if (!data.pathways || data.pathways.length === 0) {
                wrapper.innerHTML = `<div class="text-muted text-center">No mined pathways found for your allocated patients.</div>`;
                return;
            }

            wrapper.innerHTML = '';
            data.pathways.forEach(pw => {
                let stepsHtml = (pw.step_flow || []).map((s, idx) => `
                    <div class="pathway-step-card">
                        <div class="step-badge">Phase ${idx+1}</div>
                        <div class="step-title">${escapeHtml(s)}</div>
                    </div>
                `).join('<div class="step-arrow">&rarr;</div>');

                wrapper.innerHTML += `
                    <div class="pathway-story-card">
                    <div class="pathway-header">
                        <div class="pathway-title">${escapeHtml(pw.title || pw.layman_title)}</div>
                        <span class="badge-emerald">${escapeHtml(pw.Archetype_Label || 'Standard')}</span>
                    </div>
                    <div class="pathway-cohort">${escapeHtml(pw.cohort_desc || pw.layman_cohort)}</div>
                    
                    <div class="pathway-stats-grid">
                        <div class="pstat-box">
                            <div class="pstat-label">⏱️ Avg Duration</div>
                            <div class="pstat-val">${escapeHtml(pw.Average_Recovery_Time_weeks ?? '—')} Wks</div>
                        </div>
                        <div class="pstat-box">
                            <div class="pstat-label">Cohort ≥80% proxy</div>
                            <div class="pstat-val text-emerald">${pw.Success_Rate == null ? '—' : `${escapeHtml(pw.Success_Rate)}%`}</div>
                        </div>
                        <div class="pstat-box">
                            <div class="pstat-label">Mean recovery proxy</div>
                            <div class="pstat-val">${pw.Average_Recovery_Pct == null ? '—' : `${escapeHtml(pw.Average_Recovery_Pct)}%`}</div>
                        </div>
                        <div class="pstat-box">
                            <div class="pstat-label">👥 Cohort Pool</div>
                            <div class="pstat-val">${escapeHtml(pw.Sample_Size)} Patients</div>
                        </div>
                    </div>

                    <p class="pathway-story-txt">${escapeHtml(pw.story || pw.layman_story)}</p>
                    
                    <div class="pathway-flowchart-title">Recorded pathway-template steps</div>
                    <div class="pathway-step-flow">${stepsHtml || '<p>No step records are linked to this pathway template.</p>'}</div>
                    <p class="section-desc">${escapeHtml(pw.Outcome_Evidence_Scope || '')} Metric: ${escapeHtml(pw.Outcome_Metric || 'not specified')}.</p>
                </div>
                `;
            });
        } catch (err) { console.error("Error loading mined pathways:", err); }
    }

    // ----------------------------------------------------------------------
    // FULL BODY HUMAN ANATOMICAL SVG
    // ----------------------------------------------------------------------
    function renderFullBodyHumanSvg(painScore, bodyPart, compEvent, romVal, quadVal) {
        const wrapper = document.getElementById('svgPainMapWrapper');
        if (!wrapper) return;
        
        let radius = 14;
        let fillColor = '#10B981';
        let glowColor = 'rgba(16, 185, 129, 0.3)';

        if (painScore >= 8.0) { radius = 28; fillColor = '#E11D48'; glowColor = 'rgba(225, 29, 72, 0.5)'; }
        else if (painScore >= 5.0) { radius = 22; fillColor = '#F59E0B'; glowColor = 'rgba(245, 158, 11, 0.4)'; }
        else if (painScore >= 2.0) { radius = 16; fillColor = '#EAB308'; glowColor = 'rgba(234, 179, 8, 0.3)'; }

        let spotX = 220, spotY = 295;
        let partLabel = "Right Knee Joint";

        const bp = (bodyPart || "").toLowerCase();
        const inj = (currentInjuryType || "").toLowerCase();

        if (bp.includes("shoulder") || inj.includes("rotator")) { spotX = 260; spotY = 85; partLabel = "Right Shoulder"; }
        else if (bp.includes("ankle") || inj.includes("ankle") || inj.includes("achilles")) { spotX = 220; spotY = 410; partLabel = "Right Ankle Joint"; }
        else if (bp.includes("lower leg") || inj.includes("tibial")) { spotX = 220; spotY = 350; partLabel = "Right Tibia / Lower Leg"; }
        else if (bp.includes("arm") || bp.includes("elbow")) { spotX = 275; spotY = 135; partLabel = "Right Elbow Joint"; }
        else if (bp.includes("wrist") || bp.includes("hand")) { spotX = 285; spotY = 185; partLabel = "Right Wrist"; }
        else if (bp.includes("neck") || bp.includes("cervical")) { spotX = 200; spotY = 55; partLabel = "Cervical Spine (Neck)"; }
        else if (bp.includes("spine") || bp.includes("lumbar") || bp.includes("back")) { spotX = 200; spotY = 145; partLabel = "Lumbar Spine / Core"; }
        else if (bp.includes("hip") || bp.includes("pelvis")) { spotX = 220; spotY = 195; partLabel = "Right Hip Joint"; }
        else { spotX = 220; spotY = 295; partLabel = "Right Knee Joint"; }

        let compMarker = '';
        if (compEvent) {
            compMarker = `
            <g transform="translate(270, ${spotY - 20})">
                <rect x="0" y="0" width="125" height="44" rx="6" fill="#FEF2F2" stroke="#EF4444" stroke-width="1.5"/>
                <text x="62" y="17" fill="#991B1B" font-size="10" font-weight="bold" text-anchor="middle">OVERLOAD ALERT</text>
                <text x="62" y="32" fill="#7F1D1D" font-size="9" text-anchor="middle">${compEvent.Muscle_Overloaded} (${compEvent.Compensation_Level})</text>
            </g>
            <line x1="${spotX}" y1="${spotY}" x2="270" y2="${spotY}" stroke="#EF4444" stroke-width="1.5" stroke-dasharray="3,3"/>
            `;
        }

        wrapper.innerHTML = `
        <svg width="400" height="460" viewBox="0 0 400 460">
            <defs>
                <filter id="glow"><feGaussianBlur stdDeviation="6" result="blur" /><feComposite in="SourceGraphic" in2="blur" operator="over" /></filter>
            </defs>

            <ellipse cx="200" cy="35" rx="18" ry="22" fill="#E2E8F0" stroke="#94A3B8" stroke-width="1.5"/>
            <rect x="194" y="55" width="12" height="15" rx="2" fill="#CBD5E1"/>

            <path d="M 165 70 L 235 70 L 225 180 L 175 180 Z" fill="#CBD5E1" stroke="#94A3B8" stroke-width="1.5"/>
            <line x1="200" y1="70" x2="200" y2="180" stroke="#94A3B8" stroke-width="1.5" stroke-dasharray="3,3"/>

            <path d="M 165 70 L 140 85 L 125 135 L 115 185" fill="none" stroke="#CBD5E1" stroke-width="14" stroke-linecap="round"/>
            <circle cx="140" cy="85" r="7" fill="#94A3B8"/>
            <circle cx="125" cy="135" r="6" fill="#94A3B8"/>

            <path d="M 235 70 L 260 85 L 275 135 L 285 185" fill="none" stroke="#CBD5E1" stroke-width="14" stroke-linecap="round"/>
            <circle cx="260" cy="85" r="7" fill="#94A3B8"/>
            <circle cx="275" cy="135" r="6" fill="#94A3B8"/>

            <path d="M 175 180 L 225 180 L 230 210 L 170 210 Z" fill="#94A3B8"/>

            <path d="M 180 210 L 180 290 L 180 380 L 180 430" fill="none" stroke="#CBD5E1" stroke-width="18" stroke-linecap="round"/>
            <circle cx="180" cy="295" r="8" fill="#94A3B8"/>
            <circle cx="180" cy="410" r="6" fill="#94A3B8"/>

            <path d="M 220 210 L 220 290 L 220 380 L 220 430" fill="none" stroke="#CBD5E1" stroke-width="18" stroke-linecap="round"/>
            <circle cx="220" cy="295" r="8" fill="#94A3B8"/>
            <circle cx="220" cy="410" r="6" fill="#94A3B8"/>

            <circle cx="${spotX}" cy="${spotY}" r="${radius + 12}" fill="${glowColor}" filter="url(#glow)"/>
            <circle cx="${spotX}" cy="${spotY}" r="${radius + 4}" fill="none" stroke="${fillColor}" stroke-width="1.5" stroke-dasharray="4,2"/>
            <circle cx="${spotX}" cy="${spotY}" r="${radius}" fill="${fillColor}" stroke="#FFFFFF" stroke-width="2"/>
            <text x="${spotX}" y="${spotY + 4}" fill="#FFFFFF" font-size="11" font-weight="bold" text-anchor="middle">${painScore}</text>

            <rect x="15" y="15" width="155" height="42" rx="6" fill="#FFFFFF" stroke="#CBD5E1"/>
            <text x="92" y="32" fill="#0F172A" font-size="11" font-weight="bold" text-anchor="middle">${partLabel}</text>
            <text x="92" y="48" fill="#64748B" font-size="9" text-anchor="middle">ROM: ${romVal || '-'}° | Strength: ${quadVal || '-'}%</text>

            ${compMarker}
        </svg>
        `;
    }

    function renderPressurePoints(points) {
        const container = document.getElementById('pressureFlagsContainer');
        if (!container) return;
        container.innerHTML = (!points || points.length === 0) ? `<div class="alert-box-info">✓ No critical pressure point flags detected!</div>` : '';
        if (points) points.forEach(p => {
            container.innerHTML += `<div class="flag-box"><div class="flag-title">ALERT: ${p.Category} — ${p.Flag}</div><div class="flag-detail">Recorded Value: <b>${p.Actual}</b></div></div>`;
        });
    }

    function renderExpectedVsActual(eva) {
        const container = document.getElementById('expectedVsActualContainer');
        if (!container || !eva || Object.keys(eva).length === 0) return;
        let html = `<table class="data-table"><thead><tr><th>Metric</th><th>Expected</th><th>Actual</th><th>Deviation</th></tr></thead><tbody>`;
        Object.keys(eva).forEach(m => {
            const d = eva[m];
            html += `<tr><td><b>${m}</b></td><td>${d.predicted.toFixed(1)}</td><td>${d.actual.toFixed(1)}</td><td><strong style="color:${d.flagged ? '#E11D48' : '#059669'}">${d.pct_deviation >= 0 ? '+' : ''}${d.pct_deviation.toFixed(1)}%</strong></td></tr>`;
        });
        container.innerHTML = html + `</tbody></table>`;
    }

    function renderTherapyPlan(plan) {
        const container = document.getElementById('therapyPlanContainer');
        if (!container) return;
        if (!plan || plan.length === 0) {
            container.innerHTML = `<div class="text-muted text-center">No therapy plan prescribed for this week.</div>`;
            return;
        }
        let html = `<table class="data-table"><thead><tr><th>Exercise</th><th>Target</th><th>Sets x Reps</th><th>Duration</th><th>Therapist Notes</th></tr></thead><tbody>`;
        plan.forEach(ex => {
            html += `<tr><td><b>${ex.Exercise_Name}</b></td><td>${ex.Target_Muscle}</td><td>${ex.Sets} &times; ${ex.Repetitions}</td><td>${ex.Duration_min} min</td><td class="italic-txt">${ex.Therapist_Notes || '-'}</td></tr>`;
        });
        container.innerHTML = html + `</tbody></table>`;
    }

    function renderWeeklyRecalPanel(recal) {
        const panel = document.getElementById('weeklyRecalPanel');
        if (!panel) return;
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

        const tbody = document.getElementById('evrWeeklyTbody');
        const metricLabels = {
            Pain: 'Pain Score (0-10)',
            ROM_deg: 'Range of Motion (deg)',
            Strength_pct: 'Quadriceps Strength (%)',
            Balance_Score: 'Balance Score',
            Walking_Speed_mps: 'Walking Speed (m/s)'
        };
        tbody.innerHTML = '';
        const evr = recal.expected_vs_real || {};
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
            pathway ? (pathway.Archetype_Label || pathway.Pathway_Name || 'Unspecified') : 'No pathway recommendation: treatment outcome linkage is unavailable';
    }

    async function loadDoctorRoster() {
        if (!['ADMIN', 'DOCTOR'].includes(currentUserRole)) return;
        try {
            const res = await fetchWithAuth('/api/doctors');
            const data = await res.json();
            const tbody = document.getElementById('doctorRosterTbody');
            const admDocSel = document.getElementById('admAssignedDocSelect');
            const reassignDocSel = document.getElementById('reassignDoctorSelect');
            const regDocSel = document.getElementById('regAssignedDocSelect');

            if (tbody && currentUserRole === 'ADMIN') tbody.innerHTML = '';
            if (admDocSel) admDocSel.innerHTML = '';
            if (reassignDocSel) reassignDocSel.innerHTML = '';
            if (regDocSel) regDocSel.innerHTML = '';

            (data.doctors || []).forEach(d => {
                if (tbody && currentUserRole === 'ADMIN') {
                    tbody.innerHTML += `<tr>
                        <td><b>${d.Doctor_ID}</b></td>
                        <td>${d.Doctor_Name}</td>
                        <td>${d.Specialization}</td>
                        <td>${d.Hospital_Branch}</td>
                        <td>${d.Years_Experience} yrs</td>
                        <td>${d.Contact_Email}</td>
                    </tr>`;
                }
                [admDocSel, reassignDocSel, regDocSel].forEach(sel => {
                    if (sel) {
                        const opt = document.createElement('option');
                        opt.value = d.Doctor_ID;
                        opt.textContent = `${d.Doctor_Name} (${d.Doctor_ID})`;
                        sel.appendChild(opt);
                    }
                });
            });

            if (regDocSel && currentUserRole === 'DOCTOR' && currentLinkedDoc) {
                regDocSel.value = currentLinkedDoc;
            }
        } catch (err) { console.error(err); }
    }

    async function loadAdminOverview() {
        if (currentUserRole !== 'ADMIN') return;
        try {
            const res = await fetchWithAuth('/api/admin/overview');
            const data = await res.json();
            if (document.getElementById('admTotalPatients')) document.getElementById('admTotalPatients').textContent = data.total_patients;
            if (document.getElementById('admTotalDoctors')) document.getElementById('admTotalDoctors').textContent = data.total_doctors;
            if (document.getElementById('admTotalPathways')) document.getElementById('admTotalPathways').textContent = data.total_pathways;

            const reassignPatSel = document.getElementById('reassignPatientSelect');
            if (reassignPatSel) {
                const pRes = await fetchWithAuth('/api/patients');
                const pData = await pRes.json();
                reassignPatSel.innerHTML = '';
                (pData.patients || []).forEach(pid => {
                    const opt = document.createElement('option');
                    opt.value = pid; opt.textContent = pid;
                    reassignPatSel.appendChild(opt);
                });
            }
        } catch (err) { console.error(err); }
    }

    let treatmentTaxonomy = null;

    async function loadTreatmentClones() {
        if (!currentPatientId) return;

        if (!treatmentTaxonomy) {
            try {
                const resTax = await fetchWithAuth('/api/treatment/taxonomy');
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
            } catch (err) { console.error(err); }
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

        const category = document.getElementById('rxCategorySelect') ? document.getElementById('rxCategorySelect').value : 'rom';
        const exercise_type = document.getElementById('rxSubtypeSelect') ? document.getElementById('rxSubtypeSelect').value : 'rom_passive';
        const start_week = parseInt(document.getElementById('rxStartWeek') ? document.getElementById('rxStartWeek').value : 4) || 4;
        const duration_weeks = parseInt(document.getElementById('rxDurationWeeks') ? document.getElementById('rxDurationWeeks').value : 4) || 4;
        const intensity = document.getElementById('rxIntensitySelect') ? document.getElementById('rxIntensitySelect').value : 'Standard';

        try {
            const res = await fetchWithAuth(`/api/patient/${currentPatientId}/treatment/simulate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ category, exercise_type, start_week, duration_weeks, intensity })
            });
            const data = await res.json();

            const banner = document.getElementById('rxApprBanner');
            if (banner && data.appropriateness) {
                banner.textContent = data.appropriateness.message;
                if (data.appropriateness.status === 'warning') banner.className = 'flag-box mt-4';
                else if (data.appropriateness.status === 'success') {
                    banner.className = 'alert-box-info mt-4';
                    banner.style.borderLeft = '4px solid #10B981';
                } else banner.className = 'alert-box-info mt-4';
            }

            if (data.impact) {
                if (document.getElementById('rxRomImpact')) document.getElementById('rxRomImpact').textContent = `+${data.impact.rom_boost.toFixed(1)}°`;
                if (document.getElementById('rxStrengthImpact')) document.getElementById('rxStrengthImpact').textContent = `+${data.impact.strength_boost.toFixed(1)}%`;
                if (document.getElementById('rxPainImpact')) document.getElementById('rxPainImpact').textContent = `${data.impact.pain_delta > 0 ? '+' : ''}${data.impact.pain_delta.toFixed(1)}`;
                
                const disEl = document.getElementById('rxDischargeImpact');
                if (disEl) {
                    disEl.textContent = `${data.impact.discharge_weeks_new} wks (${data.impact.discharge_delta >= 0 ? '+' : ''}${data.impact.discharge_delta} wks)`;
                    disEl.style.color = data.impact.discharge_delta <= 0 ? '#10B981' : '#E11D48';
                }
            }

            if (document.getElementById('rxSummaryText')) document.getElementById('rxSummaryText').textContent = data.reaction_summary;

            renderDivergentChart(data.baseline_trajectory, data.treatment_trajectory, data.exercise.name);

        } catch (err) { console.error("Error simulating treatment:", err); }
    }

    function renderDivergentChart(baseTraj, txTraj, exerciseName) {
        const chartEl = document.getElementById('clonesChart');
        if (!chartEl) return;
        const ctx = chartEl.getContext('2d');
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
        const res = await fetchWithAuth(`/api/patient/${currentPatientId}`);
        const data = await res.json();
        const p = data.patient, pred = data.prediction;

        if (document.getElementById('realPatientInfoTable')) {
            document.getElementById('realPatientInfoTable').innerHTML = `<tr><td>Patient ID</td><td>${p.Patient_ID}</td></tr><tr><td>Age / Gender</td><td>${p.Age} / ${p.Gender}</td></tr><tr><td>Injury Type</td><td>${p.Injury_Type}</td></tr><tr><td>Body Part</td><td>${p.Body_Part}</td></tr><tr><td>Severity</td><td>${p.Severity}</td></tr><tr><td>Baseline Pain / ROM</td><td>${p.Pain_Score}/10 | ${p.Range_of_Motion_deg}°</td></tr>`;
        }
        if (document.getElementById('digitalTwinStateTable')) {
            document.getElementById('digitalTwinStateTable').innerHTML = `<tr><td>Twin ID</td><td>DT_${p.Patient_ID}</td></tr><tr><td>Predicted Recovery %</td><td class="text-emerald"><b>${pred.predicted_pct}%</b></td></tr><tr><td>Predicted Duration</td><td class="text-emerald"><b>${pred.predicted_weeks} wks</b></td></tr><tr><td>Model R² Confidence</td><td>${pred.r2_score}</td></tr>`;
        }

        let simHtml = `<table class="data-table"><thead><tr><th>Patient ID</th><th>Similarity</th><th>Age / Gender</th><th>Injury</th><th>Severity</th><th>Recovery %</th><th>Duration</th></tr></thead><tbody>`;
        data.similar_patients.forEach(sp => {
            simHtml += `<tr><td><b>${sp.Patient_ID}</b></td><td class="text-emerald"><b>${(sp.Similarity_Score * 100).toFixed(0)}%</b></td><td>${sp.Age} / ${sp.Gender}</td><td>${sp.Injury_Type}</td><td>${sp.Severity}</td><td>${sp.Final_Recovery_Pct.toFixed(0)}%</td><td>${sp.Actual_Recovery_Time_weeks.toFixed(0)} wks</td></tr>`;
        });
        if (document.getElementById('similarPatientsTableWrapper')) {
            document.getElementById('similarPatientsTableWrapper').innerHTML = simHtml + `</tbody></table>`;
        }
    }

    async function loadRecalibrationHistory() {
        if (!currentPatientId) return;
        try {
            const res = await fetchWithAuth(`/api/patient/${currentPatientId}/recalibration_history`);
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
                
                const pExp = h.Pain_Expected != null ? parseFloat(h.Pain_Expected).toFixed(1) : '-';
                const pAct = h.Pain_Actual != null ? parseFloat(h.Pain_Actual).toFixed(1) : '-';
                const rExp = h.ROM_Expected != null ? parseFloat(h.ROM_Expected).toFixed(0) : '-';
                const rAct = h.ROM_Actual != null ? parseFloat(h.ROM_Actual).toFixed(0) : '-';
                const sExp = h.Strength_Expected != null ? parseFloat(h.Strength_Expected).toFixed(0) : '-';
                const sAct = h.Strength_Actual != null ? parseFloat(h.Strength_Actual).toFixed(0) : '-';

                tbody.innerHTML += `<tr>
                    <td><b>Week ${h.Week}</b></td>
                    <td>${h.Old_Predicted_Weeks ? parseFloat(h.Old_Predicted_Weeks).toFixed(1) + ' wks' : '-'}</td>
                    <td>${h.Recalibrated_Recovery_Weeks ? parseFloat(h.Recalibrated_Recovery_Weeks).toFixed(1) + ' wks' : '-'}</td>
                    <td style="color:${deltaColor}; font-weight:700;">${deltaStr}</td>
                    <td>${pExp} / <b>${pAct}</b></td>
                    <td>${rExp}° / <b>${rAct}°</b></td>
                    <td>${sExp}% / <b>${sAct}%</b></td>
                    <td style="color:#6366F1; font-weight:600;">${h.Updated_Pathway || 'Standard'}</td>
                    <td style="font-size:0.82rem; color:#64748B;">${notes}</td>
                </tr>`;
            });
        } catch (err) { console.error(err); }
    }

    async function loadEvaluationMetrics() {
        try {
            const res = await fetchWithAuth('/api/metrics');
            const data = await res.json();

            const dataNote = document.getElementById('evaluationDataNote');
            if (dataNote) dataNote.textContent = `${data.evaluation_data_note} ${data.recovery_target_note}`;
            const baselineNote = document.getElementById('evaluationBaselineNote');
            if (baselineNote) {
                const fmt = value => Number.isFinite(value) ? Number(value).toFixed(3) : '—';
                baselineNote.textContent = `Recovery proxy: model R² ${fmt(data.metrics_pct.R2)} / RMSE ${fmt(data.metrics_pct.RMSE)}; training-mean R² ${fmt(data.baseline_metrics_pct.R2)} / RMSE ${fmt(data.baseline_metrics_pct.RMSE)}. Duration: model R² ${fmt(data.metrics_weeks.R2)} / RMSE ${fmt(data.metrics_weeks.RMSE)}; training-mean R² ${fmt(data.baseline_metrics_weeks.R2)} / RMSE ${fmt(data.baseline_metrics_weeks.RMSE)}.`;
            }

            if (document.getElementById('kpiPctR2')) document.getElementById('kpiPctR2').textContent = Number.isFinite(data.metrics_pct.R2) ? data.metrics_pct.R2.toFixed(3) : 'Unavailable (insufficient observed outcomes)';
            if (document.getElementById('kpiPctRmse')) document.getElementById('kpiPctRmse').textContent = Number.isFinite(data.metrics_pct.RMSE) ? data.metrics_pct.RMSE.toFixed(2) : '—';
            if (document.getElementById('kpiWksR2')) document.getElementById('kpiWksR2').textContent = Number.isFinite(data.metrics_weeks.R2) ? data.metrics_weeks.R2.toFixed(3) : 'Unavailable (insufficient observed outcomes)';
            if (document.getElementById('kpiWksRmse')) document.getElementById('kpiWksRmse').textContent = Number.isFinite(data.metrics_weeks.RMSE) ? data.metrics_weeks.RMSE.toFixed(2) : '—';

            const fiContainer = document.getElementById('featureImportanceList');
            if (fiContainer) {
                fiContainer.innerHTML = '';
                const maxImp = Math.max(...data.feature_importance.map(f => f.Importance));
                data.feature_importance.forEach(f => {
                    fiContainer.innerHTML += `<div class="fi-item"><div class="fi-name">${f.Feature}</div><div class="fi-bar-wrapper"><div class="fi-bar-fill" style="width: ${(f.Importance / maxImp * 100).toFixed(0)}%"></div></div><div style="font-size:0.85rem; font-weight:600; width:45px;">${(f.Importance * 100).toFixed(1)}%</div></div>`;
                });
            }
        } catch (err) { console.error(err); }
    }
});
