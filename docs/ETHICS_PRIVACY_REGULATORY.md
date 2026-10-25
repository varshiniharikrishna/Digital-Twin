# Ethics, privacy and regulatory assessment — DRAFT; requires legal/regulatory review

## Intended use and evidence boundary
This repository is a research/prototype rehabilitation analytics demonstration, not cleared/approved, not validated for clinical decisions, and not a replacement for a clinician. Intended-use language must be finalized before any filing or deployment. Bundled `output/` data are synthetic; a user-provided file is not automatically de-identified by software checks.

## Privacy and governance checklist
Before real data: establish lawful basis/authorization, consent or waiver as applicable, data-use agreements, purpose limitation, minimization, retention/deletion, access controls, encryption, breach response, data-subject rights, DPIA where required, and institutional ethics review. The adapter rejects obvious identifiers but this is not a de-identification certification. HHS HIPAA guidance describes Safe Harbor and Expert Determination methods; neither is replaced by this code. GDPR health-data processing requires analysis of Article 9 conditions and other applicable requirements. [HHS de-identification guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html); [GDPR text](https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng).

`privacy.py` provides salted pseudonymization, obvious-identifier checks, minimal audit-event writing and an optional k-anonymity diagnostic; deployment must manage the secret salt and logs under approved policy. A simple k value is not a complete privacy risk assessment.

## Fairness plan
Report missingness, calibration/error and uncertainty by sex/gender as available, age bands, injury type/region, severity and relevant intersections; include group sample counts and confidence intervals; predefine minimum sample sizes; investigate label quality, access and measurement differences; do not infer fairness from aggregate R². `reports/fairness/` records subgroup estimates only where data supports them. Synthetic subgroup results are software checks only.

## Regulatory pathways to confirm
Classification depends on intended purpose, claims, users and jurisdiction. FDA clinical decision-support analysis should apply the current FDA CDS guidance and statutory criteria; individual patient-specific predictions/recommendations may remain device functions depending on intended use. EU MDR Annex VIII Rule 11 may apply to software providing information used for diagnostic or therapeutic decisions; class depends on potential impact/risk. In India, standalone software meeting the medical-device definition can be regulated under Medical Devices Rules 2017; current CDSCO software guidance/classification must be checked. These are likely pathways to assess, not classifications or legal advice. [FDA CDS guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software); [EU MDR](https://eur-lex.europa.eu/eli/reg/2017/745/oj/eng); [CDSCO Medical Devices Rules](https://cdsco.gov.in/opencms/resources/UploadCDSCOWeb/2022/m_device/Medical%20Devices%20Rules%2C%202017.pdf).

## Open items
No real clinical validation, prospective utility evidence, clinical risk management file, legal determination, DPIA, ethics approval, or regulatory consultation is present. Expert review is required before external use.
