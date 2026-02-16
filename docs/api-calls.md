# API Reference -- External Endpoints

All HTTP calls use `hishel.AsyncCacheClient` (SQLite-backed disk cache, 24 h TTL)
via [`src/data/http.py`](../src/data/http.py).  GET and POST are both cached.
Every client retries 3 times with exponential back-off (1-10 s) on
`ConnectError` / `ReadTimeout` / `HTTPStatusError`.

> **Test every curl below** before assuming it still works -- biomedical
> APIs change schemas without warning.

---

## 1. Drug Normalizer (`src/data/normalizer.py`)

Resolves a free-text drug name to a canonical name, PubChem CID,
ChEMBL ID, and synonyms.  Uses PubChem first, falls back to ChEMBL.

### 1a. PubChem -- name to CID

```bash
curl -H "Accept: application/json" \
  "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/cids/JSON"
```

**Response** (JSON):
```json
{"IdentifierList": {"CID": [2244]}}
```

### 1b. PubChem -- CID to synonyms

```bash
curl -H "Accept: application/json" \
  "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/synonyms/JSON"
```

**Response** (JSON): `InformationList.Information[0].Synonym` is a list
of strings.

### 1c. PubChem -- CID cross-references (ChEMBL fallback)

```bash
curl -H "Accept: application/json" \
  "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/xrefs/RegistryID/JSON"
```

**Response** (JSON): `InformationList.Information[0].RegistryID` -- scan
for entries matching `CHEMBL\d+`.

### 1d. ChEMBL -- molecule search

```bash
curl -H "Accept: application/json" \
  "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q=aspirin&limit=5"
```

**Response** (JSON): `molecules[*].molecule_chembl_id`.

---

## 2. OpenTargets (`src/data/opentargets.py`)

GraphQL endpoint.  Two query types share the same URL.

### 2a. Known drugs (drug-target-disease associations)

```bash
curl -X POST "https://api.platform.opentargets.org/api/v4/graphql" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query($chemblId: String!) { drug(chemblId: $chemblId) { name knownDrugs(size: 50) { rows { target { id approvedSymbol } disease { id name } phase status references { source ids } } } } }",
    "variables": {"chemblId": "CHEMBL25"}
  }'
```

**Response** (JSON): `data.drug.knownDrugs.rows[]` -- each row has
`target`, `disease`, `phase`, `status`, `references`.

### 2b. Text-mining evidence (per target-disease pair)

```bash
curl -X POST "https://api.platform.opentargets.org/api/v4/graphql" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query($efoId: String!, $ensemblId: String!) { disease(efoId: $efoId) { evidences(ensemblIds: [$ensemblId], datasourceIds: [\"europepmc\"], size: 10) { rows { score literature textMiningSentences { dEnd dStart tEnd tStart section text } } } } }",
    "variables": {"efoId": "EFO_0000311", "ensemblId": "ENSG00000073756"}
  }'
```

**Response** (JSON): `data.disease.evidences.rows[]` -- each row has
`score`, `literature` (PMID list), `textMiningSentences`.

---

## 3. DGIdb (`src/data/dgidb.py`)

GraphQL endpoint.

### 3a. Drug-gene interactions

```bash
curl -X POST "https://dgidb.org/api/graphql" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query($names: [String!]!) { drugs(names: $names) { nodes { name conceptId interactions { interactionScore interactionTypes { type directionality } interactionAttributes { name value } gene { name conceptId } publications { pmid } sources { sourceDbName fullName } } } } }",
    "variables": {"names": ["ASPIRIN"]}
  }'
```

**Response** (JSON): `data.drugs.nodes[].interactions[]` -- each
interaction has `gene`, `interactionTypes`, `publications`,
`interactionScore`, `sources`.

---

## 4. PubChem (`src/data/pubchem.py`)

PUG-REST endpoints.  Timeout 60 s (classification responses can be very
large for some CIDs).

### 4a. Name to CID

Same as [1a](#1a-pubchem----name-to-cid) but without explicit headers.

```bash
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/cids/JSON"
```

### 4b. Pharmacological classifications

```bash
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/classification/JSON"
```

**Response** (JSON): `Hierarchies.Hierarchy[]` -- capped to first 20
hierarchies, each with `SourceName` and `Node[]` entries.

> **Warning**: this endpoint can return 80+ MB for broadly classified
> compounds and occasionally uses non-UTF-8 bytes (e.g. `0xAE`).  The
> client falls back to `latin-1` decoding.

### 4c. Bioassay summary

```bash
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/assaysummary/JSON"
```

**Response** (JSON): `Table.Columns.Column` (list of header names) +
`Table.Row[]` (list of row dicts with `Cell` arrays).

---

## 5. PharmGKB (`src/data/pharmgkb.py`)

REST API.  All endpoints require `Accept: application/json`.

### 5a. Drug search (resolve name to PharmGKB ID)

```bash
curl -H "Accept: application/json" \
  "https://api.pharmgkb.org/v1/data/chemical?name=aspirin"
```

**Response** (JSON): `data[0].id` (e.g. `PA448497`).

### 5b. Clinical annotations

```bash
curl -H "Accept: application/json" \
  "https://api.pharmgkb.org/v1/data/clinicalAnnotation?relatedChemicals.accessionId=PA448497"
```

**Response** (JSON): `data[]` -- each annotation has `evidenceLevel`,
`relatedDiseases`, `relatedGenes`, `relatedChemicals`.

### 5c. Drug label annotations

```bash
curl -H "Accept: application/json" \
  "https://api.pharmgkb.org/v1/data/drugLabel?relatedChemicals.accessionId=PA448497"
```

**Response** (JSON): `data[]` -- each label has `name`, `source`,
`relatedDiseases`, `relatedGenes`.

---

## 6. ChEMBL (`src/data/chembl.py`)

REST API.

### 6a. Mechanisms of action

```bash
curl -H "Accept: application/json" \
  "https://www.ebi.ac.uk/chembl/api/data/mechanism.json?molecule_chembl_id=CHEMBL25&limit=20"
```

**Response** (JSON): `mechanisms[]` -- each has
`mechanism_of_action`, `target_chembl_id`, `action_type`.

### 6b. Binding / functional activities

```bash
curl -H "Accept: application/json" \
  "https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id=CHEMBL25&standard_type__in=IC50,EC50,Ki,Kd&pchembl_value__isnull=false&order_by=-pchembl_value&limit=20"
```

**Response** (JSON): `activities[]` -- each has `target_chembl_id`,
`target_pref_name`, `pchembl_value`, `standard_type`,
`standard_value`, `standard_units`, `assay_description`.

---

## 7. PubMed (`src/data/pubmed.py`)

NCBI E-utilities.  Replace `user@example.com` with your
`Settings.entrez_email`.

### 7a. eSearch (find PMIDs by query)

```bash
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=aspirin+AND+(mechanism+OR+therapeutic+OR+treatment)&retmax=20&retmode=json&sort=relevance&email=user@example.com"
```

**Response** (JSON): `esearchresult.idlist` -- list of PMID strings.

### 7b. eFetch (retrieve abstracts by PMID batch)

```bash
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=35261515,25701173&rettype=abstract&retmode=xml&email=user@example.com"
```

**Response** (XML): `PubmedArticleSet` -- each `PubmedArticle` has
`MedlineCitation.PMID`, `MedlineCitation.Article.ArticleTitle`,
`MedlineCitation.Article.Abstract.AbstractText`.

Batched at 50 PMIDs per request.

---

## 8. CTD (`src/data/ctd.py`)

Bulk download.  120 s timeout.

### 8a. Chemicals-diseases associations

```bash
curl -o CTD_chemicals_diseases.tsv.gz \
  "http://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz"
```

**Response**: gzipped TSV (~180 MB compressed).  Columns:
`ChemicalName`, `ChemicalID`, `CasRN`, `DiseaseName`, `DiseaseID`,
`DirectEvidence`, `InferenceGeneSymbol`, `InferenceScore`, `OmimIDs`,
`PubMedIDs`.

Header lines start with `#`.  Filtered to rows where
`DirectEvidence` contains `"therapeutic"`.

---

## Summary

| # | Client | Base URL | Method | Endpoints | Format |
|---|--------|----------|--------|-----------|--------|
| 1 | Normalizer (PubChem) | `pubchem.ncbi.nlm.nih.gov/rest/pug` | GET | 3 | JSON |
| 1 | Normalizer (ChEMBL) | `www.ebi.ac.uk/chembl/api/data` | GET | 1 | JSON |
| 2 | OpenTargets | `api.platform.opentargets.org/api/v4/graphql` | POST | 2 queries | JSON |
| 3 | DGIdb | `dgidb.org/api/graphql` | POST | 1 query | JSON |
| 4 | PubChem | `pubchem.ncbi.nlm.nih.gov/rest/pug` | GET | 3 | JSON |
| 5 | PharmGKB | `api.pharmgkb.org/v1/data` | GET | 3 | JSON |
| 6 | ChEMBL | `www.ebi.ac.uk/chembl/api/data` | GET | 2 | JSON |
| 7 | PubMed | `eutils.ncbi.nlm.nih.gov/entrez/eutils` | GET | 2 | JSON + XML |
| 8 | CTD | `ctdbase.org/reports` | GET | 1 | gzip TSV |

**Total: 15 distinct API calls across 8 clients.**
