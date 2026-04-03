# Documentation technique du dépôt `casablanca-builtup-pipeline`

## Table des matières

1. [Périmètre et objectif du dépôt](#1-périmètre-et-objectif-du-dépôt)
2. [Vue d’ensemble fonctionnelle](#2-vue-densemble-fonctionnelle)
3. [Structure utile du dépôt](#3-structure-utile-du-dépôt)
4. [Fichiers de configuration et paramètres structurants](#4-fichiers-de-configuration-et-paramètres-structurants)
5. [Pipeline built-up amplitude / Sentinel-2](#5-pipeline-built-up-amplitude--sentinel-2)
6. [Pipeline PSI Sentinel-1 SLC / SNAP / StaMPS](#6-pipeline-psi-sentinel-1-slc--snap--stamps)
7. [Acquisition des données et logique de sélection](#7-acquisition-des-données-et-logique-de-sélection)
8. [SNAP dans ce dépôt](#8-snap-dans-ce-dépôt)
9. [StaMPS dans ce dépôt](#9-stamps-dans-ce-dépôt)
10. [Sorties, artefacts et structure des runs](#10-sorties-artefacts-et-structure-des-runs)
11. [Gestion des erreurs, reprise et robustesse](#11-gestion-des-erreurs-reprise-et-robustesse)
12. [Commandes réelles d’exécution](#12-commandes-réelles-dexécution)
13. [Pipeline minimal vs pipeline complet](#13-pipeline-minimal-vs-pipeline-complet)
14. [Pré-requis opérationnels et conseils d’exploitation](#14-pré-requis-opérationnels-et-conseils-dexploitation)
15. [Limites visibles dans le code](#15-limites-visibles-dans-le-code)

## 1. Périmètre et objectif du dépôt

Cette documentation a été rédigée à partir du dépôt réellement inspecté dans l’environnement courant, situé sous `/Users/tahaelouali/casablanca-builtup-pipeline`.

Ce dépôt contient en réalité **deux chaînes de traitement distinctes mais liées** :

1. **Un pipeline de détection de changement bâti par amplitude**, centré sur Sentinel-1 RTC et affiné par Sentinel-2.
2. **Un pipeline PSI** fondé sur Sentinel-1 SLC, SNAP et StaMPS, destiné à produire des points et polygones candidats plus structurels.

Le nom du dépôt peut laisser croire qu’il n’existe qu’un seul pipeline. Ce n’est pas le cas. Le code inspecté montre bien :

- une arborescence `src/casablanca_builtup/` pour le pipeline amplitude/S2 ;
- une arborescence `src/casablanca_psi/` pour le pipeline PSI.

Le problème traité est la **détection de nouveaux bâtiments ou de nouvelles zones bâties probables** sur Casablanca. Dans ce dépôt, cette expression n’a pas exactement le même sens selon la chaîne utilisée :

- dans le pipeline **built-up**, on détecte d’abord un **changement de rétrodiffusion radar** cohérent avec de l’urbanisation, puis on le raffine avec des indices optiques Sentinel-2 ; le résultat final est une **zone bâtie probable**, pas une empreinte bâtiment garantie ;
- dans le pipeline **PSI**, on cherche à identifier des **points PSI émergents ou stables** et à les agréger en **candidats de nouveaux bâtiments** ; cette branche est plus proche d’une confirmation structurelle, mais l’implémentation actuelle reste partiellement scaffoldée côté exploitation scientifique des sorties StaMPS.

## 2. Vue d’ensemble fonctionnelle

### 2.1 Ce que fait globalement le dépôt

Le dépôt assemble plusieurs briques :

- **Python** pour l’orchestration, la configuration, l’acquisition, le contrôle des artefacts, le parsing et l’export ;
- **Sentinel-1** comme source radar principale ;
- **Sentinel-2** comme source optique de support dans la chaîne built-up, et comme brique potentielle mais non branchée dans l’orchestrateur PSI actuel ;
- **SNAP** pour préparer une pile SLC coregistrée et exporter un jeu d’entrée compatible StaMPS ;
- **StaMPS** pour le calcul PSI ;
- des **rapports JSON**, des **logs**, des **manifests gelés** et des **artefacts intermédiaires sur disque** pour assurer la reproductibilité et la reprise.

### 2.2 Différence entre les deux approches présentes dans le dépôt

#### Chaîne built-up

Schéma réel :

```text
Planetary Computer STAC
    -> manifests gelés
    -> composites Sentinel-1 RTC avant / après
    -> détection de changement S1
    -> composites Sentinel-2 avant / après
    -> support optique et raffinement
    -> cumul temporel
    -> densité / zones
    -> polygonisation
    -> exports vecteur
```

Cette chaîne produit des **polygones de zones bâties probables**.

#### Chaîne PSI

Schéma réel :

```text
CDSE STAC + CDSE OData + CDSE S3
    -> manifests de stacks SLC
    -> téléchargement des ZIP SLC
    -> SNAP (préparation, coregistration, export StaMPS)
    -> StaMPS
    -> export CSV des points PSI
    -> parsing Python
    -> fusion des points en polygones
    -> export final
```

Cette chaîne vise des **points PSI** puis des **polygones candidats**.

### 2.3 Rôle des technologies principales

- **Sentinel-1 RTC** : utilisé dans la chaîne built-up pour un traitement amplitude déjà géocodé.
- **Sentinel-1 SLC** : utilisé dans la chaîne PSI. Le format SLC est indispensable pour SNAP/StaMPS car il conserve la phase complexe.
- **Sentinel-2 L2A** : utilisé comme couche de support optique dans la chaîne built-up. Des modules PSI existent aussi pour l’optique, mais ils ne sont pas appelés par l’orchestrateur PSI actuel.
- **SNAP** : prépare les produits SLC, applique les orbites, découpe les swaths, coregistre la pile et prépare l’export vers StaMPS.
- **StaMPS** : calcule les points PSI à partir de l’export SNAP.
- **Python** : orchestre les étapes, gère les manifests, les téléchargements, la reprise, les rapports, la fusion et les exports.

## 3. Structure utile du dépôt

Les chemins ci-dessous sont relatifs à la racine du dépôt.

```text
configs/
  casablanca_city.yaml
  psi_casablanca_slc.yaml
  psi_casablanca_slc_minimal.yaml

docs/
  architecture.md
  operations.md
  psi-architecture.md

resources/
  snap_graphs/
    prepare_slc_stack.xml
    coregister_stack.xml
    stamps_export.xml
  stamps/
    export_ps_points.m

src/
  casablanca_builtup/
    cli.py
    pipeline.py
    config.py
    acquisition/stac.py
    s1/
    s2/
    fusion.py
    postprocess/vectorize.py
    resume.py
    run_context.py
    qa.py
    io.py
    evaluation.py
  casablanca_psi/
    cli.py
    pipeline.py
    config.py
    acquisition.py
    manifests.py
    snap.py
    stamps.py
    psi_results.py
    fusion.py
    export.py
    amplitude.py
    s2_refinement.py
    run_context.py

tests/
  integration/
  psi/

runs/
runs_psi/
```

### 3.1 Rôle des dossiers importants

- `configs/` : tous les fichiers YAML de pilotage.
- `docs/` : documentation d’architecture et d’exploitation déjà présente.
- `resources/snap_graphs/` : graphes XML réellement appelés par SNAP.
- `resources/stamps/` : script MATLAB/StaMPS d’export des points.
- `src/casablanca_builtup/` : pipeline amplitude / Sentinel-2.
- `src/casablanca_psi/` : pipeline PSI.
- `tests/` : tests unitaires et tests d’intégration.
- `runs/` : sorties du pipeline built-up.
- `runs_psi/` : sorties du pipeline PSI.

### 3.2 Modules structurants côté built-up

- `cli.py` : commandes Typer.
- `pipeline.py` : orchestration complète par stage.
- `acquisition/stac.py` : interrogation STAC Planetary Computer et gel des manifests.
- `s1/composite.py` : construction des composites Sentinel-1.
- `s1/detection.py` : détection de changement radar.
- `s2/composite.py` : construction des composites optiques et indices.
- `s2/refinement.py` : score de support optique.
- `fusion.py` : raffinement souple des candidats radar.
- `postprocess/vectorize.py` : cumul, densité, mask, polygonisation.
- `resume.py` : chargement/sauvegarde/réutilisation des artefacts intermédiaires.
- `run_context.py` : structure des runs et des attempts.
- `qa.py` : rapport d’exécution et métriques de synthèse.

### 3.3 Modules structurants côté PSI

- `cli.py` : commandes stage par stage.
- `pipeline.py` : orchestration PSI complète.
- `acquisition.py` : sélection des scènes SLC, auth CDSE, téléchargements OData/S3 robustes.
- `manifests.py` : structures `SlcScene` et `StackManifest`.
- `snap.py` : exécution des graphes SNAP.
- `stamps.py` : préparation StaMPS et lancement MATLAB/Octave.
- `psi_results.py` : lecture des points exportés et détection des points émergents.
- `fusion.py` : transformation de points PSI en polygones candidats.
- `export.py` : écriture des livrables PSI.

## 4. Fichiers de configuration et paramètres structurants

## 4.1 Configuration built-up : `configs/casablanca_city.yaml`

Ce fichier pilote la chaîne amplitude/Sentinel-2.

### AOI

```yaml
aoi:
  name: casablanca-city
  bbox: [-7.80, 33.40, -7.45, 33.65]
  crs: EPSG:4326
```

Le dépôt supporte soit `aoi.path`, soit `aoi.bbox`, mais pas les deux en même temps. C’est validé dans `casablanca_builtup.config.AOIConfig`.

### Fenêtres temporelles

Le pipeline built-up travaille par **périodes**. Chaque période contient :

- une fenêtre `before`
- une fenêtre `after`

Dans le fichier inspecté :

```yaml
periods:
  - id: "2023_2025"
    before:
      start: "2023-06-01"
      end: "2023-09-30"
    after:
      start: "2025-06-01"
      end: "2025-09-30"
```

### Sentinel-1

```yaml
sentinel1:
  collection: sentinel-1-rtc
  platform: SENTINEL-1A
  instrument_mode: IW
  polarizations: [vv, vh]
```

Cette branche travaille sur **RTC**, pas sur SLC.

### Sentinel-2

Le bloc `sentinel2` définit :

- la collection ;
- le seuil de couverture nuageuse ;
- la table de correspondance des bandes ;
- les classes SCL considérées comme claires ;
- les seuils d’indices optiques ;
- la stratégie de dégradation en l’absence de S2 (`allow_unavailable: true`).

### Détection, densité, polygonisation, export

Les blocs `detection`, `density`, `polygonization` et `export` pilotent :

- les seuils radar ;
- la densification spatiale ;
- les filtres géométriques ;
- le format des rasters et vecteurs de sortie.

### Cache et run

```yaml
cache:
  reuse_manifests: true
  overwrite: false

run:
  output_root: runs
  log_level: INFO
  write_log_file: true
```

`overwrite: false` est la clé du comportement de reprise : si les artefacts valides existent déjà, le pipeline les recharge au lieu de les recalculer.

## 4.2 Configuration PSI complète : `configs/psi_casablanca_slc.yaml`

Ce fichier pilote la chaîne SLC/SNAP/StaMPS.

### Acquisition

Le bloc `acquisition` contient les éléments essentiels :

- `s1_catalog_url`: `https://stac.dataspace.copernicus.eu/v1`
- `s1_collection`: `sentinel-1-slc`
- `odata_catalog_url`: `https://catalogue.dataspace.copernicus.eu/odata/v1`
- `time_window`: fenêtre globale de recherche des acquisitions SLC
- `download_transport`: `auto`, `s3` ou `odata`
- `slc_asset_priority`: priorité entre `product` et `safe_manifest`

### Authentification et credentials

Le code supporte côté PSI :

- `CDSE_ACCESS_TOKEN`
- `ACCESS_TOKEN`
- `REFRESH_TOKEN`
- `CDSE_USERNAME`
- `CDSE_PASSWORD`
- `CDSE_S3_ACCESS_KEY`
- `CDSE_S3_SECRET_KEY`
- `CDSE_S3_ENDPOINT`
- `CDSE_S3_FALLBACK_ENDPOINTS`
- `CDSE_S3_BUCKET`

La logique réelle est la suivante :

- si le transport S3 est possible et que les credentials S3 sont présents, le pipeline télécharge par S3 ;
- sinon il bascule vers OData, avec token Bearer, refresh token ou couple username/password.

### S3 / CDSE

Le bloc `acquisition.s3` définit :

- l’endpoint principal ;
- le bucket ;
- la région logique ;
- les noms des variables d’environnement ;
- les paramètres de retry bas niveau.

Le code normalise aussi les endpoints et supporte une liste d’endpoints de secours. L’implémentation actuelle utilise en pratique :

- `https://eodata.dataspace.copernicus.eu`
- `https://eodata.ams.dataspace.copernicus.eu`

### DEM

Le pipeline PSI attend un DEM local :

```yaml
dem:
  path: data/reference/dem/copdem_30m_casablanca.tif
```

SNAP vérifie explicitement que ce fichier existe avant de démarrer.

### Stacks

Chaque stack définit :

- un identifiant ;
- la direction orbitale ;
- le numéro de relative orbit ;
- les swaths `IW1/IW2/IW3` ;
- la polarisation ;
- une date maître ;
- un nombre minimal de scènes.

Dans la configuration complète, deux piles sont prévues :

- `asc_rel147_vv`
- `desc_rel154_vv`

### SNAP, StaMPS, PSI, fusion, export

Le reste du fichier pilote :

- le chemin du binaire `gpt` SNAP ;
- le répertoire des graphes XML ;
- les options Java ;
- le chemin d’installation StaMPS ;
- le binaire MATLAB ou Octave ;
- le script d’export StaMPS ;
- les seuils PSI ;
- les poids de fusion ;
- les formats de sortie ;
- les options de cache.

## 4.3 Configuration PSI minimale : `configs/psi_casablanca_slc_minimal.yaml`

Ce fichier est une variante opérationnelle dédiée au **plus petit stack PSI fonctionnel** :

- une seule pile ;
- direction `ascending` ;
- relative orbit `147` ;
- `min_scenes: 5` ;
- `scene_limit: 5` ;
- fenêtre temporelle réduite à `2023-08-01` / `2023-11-30`.

Ce mode minimal sert à valider la chaîne externe SLC -> SNAP -> StaMPS avec le plus petit jeu de données compatible.

## 5. Pipeline built-up amplitude / Sentinel-2

## 5.1 Objectif réel de la chaîne built-up

Cette chaîne n’essaie pas de reconstruire des bâtiments individuels. Elle détecte des **zones de changement radar compatibles avec de l’urbanisation**, puis les consolide avec Sentinel-2.

Le signal principal est radar. Sentinel-2 agit comme **support** et non comme veto strict.

## 5.2 Point d’entrée réel

L’entrée principale est `src/casablanca_builtup/cli.py`.

Commandes disponibles :

- `casablanca-builtup acquire-data`
- `casablanca-builtup build-composites`
- `casablanca-builtup detect-s1`
- `casablanca-builtup refine-s2`
- `casablanca-builtup polygonize`
- `casablanca-builtup export`
- `casablanca-builtup run-pipeline`
- `casablanca-builtup evaluate-run`

Chaque commande stage par stage relance `run_pipeline(..., stop_after=<stage>)`. Autrement dit, une exécution partielle **rejoue la chaîne depuis le début jusqu’au stage demandé**, en réutilisant les artefacts déjà présents.

## 5.3 Flux d’exécution réel

### Étape 1 : initialisation du run

Dans `casablanca_builtup.pipeline.execute_pipeline` :

- chargement du YAML ;
- création du `RunContext` ;
- création des dossiers de run ;
- configuration du logging ;
- écriture de `reports/resolved_config.yaml`.

### Étape 2 : AOI et grille

Le pipeline :

- charge l’AOI via `load_aoi_frame` ;
- construit une grille alignée via `build_grid` ;
- écrit `reports/grid.json`.

La grille de la configuration inspectée est en `EPSG:32629` à `10 m`.

### Étape 3 : acquisition STAC et manifests

Le module `casablanca_builtup.acquisition.stac` interroge le **Planetary Computer STAC** :

- endpoint : `https://planetarycomputer.microsoft.com/api/stac/v1`
- Sentinel-1 : collection `sentinel-1-rtc`
- Sentinel-2 : collection `sentinel-2-l2a`

Le pipeline gèle ensuite les résultats sous forme de manifests JSON par période. Ces manifests sont réutilisables si `cache.reuse_manifests` est actif.

### Étape 4 : composites Sentinel-1

`casablanca_builtup.s1.composite.build_s1_composite` charge les items STAC signés via `odc.stac.load`, puis calcule un **médian temporel** pour les bandes VV et VH.

Artefacts produits :

- `<period>_s1_before_vv.tif`
- `<period>_s1_before_vh.tif`
- `<period>_s1_after_vv.tif`
- `<period>_s1_after_vh.tif`

### Étape 5 : détection de changement Sentinel-1

`casablanca_builtup.s1.detection.detect_s1_change` calcule :

- une statistique de type likelihood-ratio sur VV ;
- les ratios VV et VH ;
- un masque candidat.

Le masque candidat résulte des seuils de `detection` :

- `alpha`
- `vv_ratio_min`
- `vh_ratio_min`
- `min_connected_pixels`
- `closing_radius_m`

Artefacts produits :

- `<period>_s1_pvalue_vv.tif`
- `<period>_s1_ratio_vv.tif`
- `<period>_s1_ratio_vh.tif`
- `<period>_s1_candidate.tif`

### Étape 6 : composites et support Sentinel-2

Si des scènes S2 existent, `casablanca_builtup.s2.composite.build_s2_composite` calcule :

- les bandes de base ;
- `NDVI`
- `NDBI`
- `MNDWI`
- éventuellement `BSI`
- `clear_count`
- `valid_count`
- `clear_fraction`
- `valid_fraction`

Des stacks intermédiaires par bande sont écrits dans `staging/s2/<period>/<phase>/`.

Ensuite, `casablanca_builtup.s2.refinement.build_s2_support` produit :

- un score de support ;
- un masque de fiabilité.

Si S2 est indisponible et que `allow_unavailable: true`, le pipeline ne s’arrête pas. Il construit un support artificiel via `build_unavailable_s2_support_like` et conserve le signal S1 seul. C’est un comportement réel, couvert par les tests.

### Étape 7 : fusion douce

`casablanca_builtup.fusion.apply_soft_refinement` combine :

- le candidat radar ;
- le support optique ;
- la fiabilité du support ;
- une éventuelle logique de **strong override** radar.

Le raster de décision est codé avec les labels définis dans `casablanca_builtup.qa.FUSION_DECISION_LABELS` :

- `not_candidate`
- `kept_s1_s2_unreliable`
- `kept_s1_s2_supported`
- `kept_strong_s1_override`
- `dropped_s2_reliable_unsupported`

Artefacts produits :

- `<period>_s2_score.tif`
- `<period>_s2_reliable.tif`
- `<period>_fusion_decision.tif`
- `<period>_refined_mask.tif`

### Étape 8 : cumul, densité, zones et polygonisation

`casablanca_builtup.postprocess.vectorize` enchaîne :

- `build_cumulative_first_change`
- `build_density_zone_mask`
- `polygonize_mask`

Artefacts produits :

- `cumulative_raw.tif`
- `cumulative_refined.tif`
- `zone_density.tif`
- `zone_mask.tif`
- `vectors/zone_polygons.parquet`

### Étape 9 : export final

Le stage `export` écrit les polygones finaux dans les formats demandés. Dans la configuration inspectée :

- `vectors/builtup_zones.parquet`
- `vectors/builtup_zones.gpkg`

## 5.4 Rapports et métriques built-up

Le pipeline enregistre dans `reports/run_report.json` :

- l’identifiant du run ;
- la configuration utilisée ;
- la grille ;
- l’état des stages ;
- le détail par période ;
- des métriques comme le nombre de pixels candidats, les surfaces, le nombre de polygones et les histogrammes de décision.

## 6. Pipeline PSI Sentinel-1 SLC / SNAP / StaMPS

## 6.1 Objectif réel de la chaîne PSI

La chaîne PSI cherche à partir d’une **pile Sentinel-1 SLC cohérente** pour produire :

1. des exports StaMPS ;
2. des points PSI ;
3. des points émergents filtrés ;
4. des polygones candidats agrégés.

Le chaînage réel présent dans le code est :

```text
validate-provider (optionnel)
    -> acquire
    -> download-slc
    -> run-snap
    -> run-stamps
    -> parse-psi
    -> fuse
    -> export
```

## 6.2 Point d’entrée réel

L’entrée principale est `src/casablanca_psi/cli.py`.

Commandes disponibles :

- `casablanca-psi validate-provider`
- `casablanca-psi run-pipeline`
- `casablanca-psi acquire`
- `casablanca-psi download-slc`
- `casablanca-psi run-snap`
- `casablanca-psi run-stamps`
- `casablanca-psi parse-psi`
- `casablanca-psi fuse`
- `casablanca-psi evaluate`

Point important : la commande `fuse` exécute la chaîne jusqu’au stage `fuse`, **puis s’arrête avant `export`**. L’export final n’est exécuté que par `run-pipeline`, ou par un rerun complet qui va au-delà du stage `fuse`.

## 6.3 Signification des notions orbit direction, relative orbit, polarization, IW, SLC

- **orbit direction** : sens de passage du satellite, `ascending` ou `descending`.
- **relative orbit** : identifiant de trace répétée. C’est un filtre essentiel pour construire une pile cohérente.
- **polarization** : canal radar utilisé, par exemple `VV` ou `VH`.
- **IW** : mode `Interferometric Wide Swath`, mode d’acquisition standard de Sentinel-1 pour ce type d’usage.
- **SLC** : `Single Look Complex`, produit complexe nécessaire à la phase, donc à l’interférométrie et à la PSI.

## 6.4 Acquisition PSI : sélection des scènes et génération des manifests

Le module `casablanca_psi.acquisition` interroge le **STAC CDSE**.

### Validation fournisseur

`validate_s1_slc_provider_fields()` :

- requête un premier item SLC ;
- affiche le mapping de propriétés trouvé ;
- affiche l’asset sélectionné.

Cette commande sert surtout à vérifier que le provider expose bien les champs attendus.

### Recherche des scènes

`query_s1_slc_stack()` :

- découpe la fenêtre temporelle en blocs de 45 jours ;
- relance la recherche jusqu’à 3 fois sur erreur provider ;
- filtre les items selon :
  - la direction ;
  - la relative orbit ;
  - le mode `IW` ;
  - le type de produit `SLC` ;
  - le niveau de traitement `L1` quand il est renseigné ;
  - la polarisation demandée.

Le code **n’effectue pas de vérification explicite de cohérence burst par burst lors de l’acquisition**. Les `iw_swaths` sont surtout consommés plus tard par SNAP.

### Sélection déterministe des scènes

Une fois les scènes filtrées, `_select_stack_scenes()` applique :

- un contrôle `min_scenes` ;
- éventuellement `scene_limit` ;
- si `master_date` existe, une fenêtre déterministe centrée sur le maître ;
- sinon les premières scènes de la pile.

### Manifests

Chaque pile devient un `StackManifest` JSON dans `manifests/<stack_id>.json` avec :

- `stack_id`
- `direction`
- `relative_orbit`
- `product_type`
- `scene_count`
- la liste des `SlcScene`

## 6.5 Téléchargement des ZIP SLC

Le téléchargement PSI est aujourd’hui la partie la plus sophistiquée du dépôt.

### Transports réels

Le code supporte trois modes :

- `download_transport: s3`
- `download_transport: odata`
- `download_transport: auto`

En mode `auto`, la logique réelle privilégie **S3** si les credentials S3 sont disponibles.

### S3 CDSE

Le client S3 est créé avec :

- `endpoint_url` explicite ;
- `signature_version="s3v4"` ;
- `botocore.config.Config` avec retries, timeouts, pool de connexions et `addressing_style="path"` ;
- un endpoint principal et des endpoints de secours.

La résolution du chemin objet suit l’ordre suivant :

1. `s3_path` déjà présent dans le manifest ;
2. cache mémoire ;
3. dérivation à partir du nom produit et de la date ;
4. href S3 alternatif dans le STAC ;
5. lookup OData.

### Téléchargement robuste de gros SAFE

Le téléchargement S3 ne télécharge plus le SAFE entier en mémoire ni en un seul flux jetable. L’implémentation actuelle :

1. liste les objets du SAFE dans le bucket ;
2. crée un répertoire de staging `<scene>.zip.parts/` ;
3. télécharge chaque membre du SAFE dans un `.part` local ;
4. reprend les membres interrompus via `Range`;
5. assemble le ZIP final seulement quand tous les membres sont complets ;
6. valide la cohérence du ZIP assemblé ;
7. renomme atomiquement `<scene>.zip.part` en `<scene>.zip`.

Ce design préserve la progression déjà téléchargée même en cas de `ResponseStreamingError` ou `IncompleteRead(...)`.

### OData

Le fallback OData télécharge le produit dans `<scene>.zip.part`, puis renomme atomiquement vers le ZIP final. Cette voie est plus simple et ne dispose pas de la granularité de reprise du staging par membre S3.

### Réutilisation et reprise

`download_stack_scenes()` :

- réutilise les ZIP déjà terminés si `cache.reuse_downloads` est actif ;
- supprime les `.zip.part` de niveau scène qui sont incomplets ;
- préserve les `.zip.parts/.../*.part` de niveau membre S3 tant qu’aucune corruption n’est prouvée.

## 6.6 Prétraitement SNAP

Le module `casablanca_psi.snap.SnapGraphRunner` vérifie d’abord :

- que `gpt_path` pointe bien vers **SNAP GPT** ;
- que le DEM existe ;
- que l’environnement Java/SNAP est exploitable.

### Séquence réelle

Pour chaque pile :

1. sélection de l’image maître à partir de `master_date` ;
2. vérification qu’il existe au moins 4 secondaires ;
3. préparation du maître et des secondaires pour chaque swath `IW1`, `IW2`, `IW3` ;
4. coregistration maître/secondaire ;
5. calcul des produits interférométriques ;
6. export StaMPS.

Le runner réutilise les sorties SNAP si `reuse_snap_outputs` est actif et si le trio `rslc`, `diff0`, `geo` existe déjà dans le répertoire d’export.

## 6.7 Que fait réellement Python avant et après SNAP

Avant SNAP :

- interrogation CDSE ;
- constitution des manifests ;
- téléchargement et reprise des ZIP SLC ;
- sélection du maître et des secondaires.

Après SNAP :

- lancement StaMPS ;
- lecture du CSV exporté ;
- filtrage des points PSI ;
- fusion en polygones ;
- export.

Il n’existe pas de branche PSI réellement orchestrée qui calcule aujourd’hui une amplitude radar avant/après ni un support optique S2 à l’intérieur de `casablanca_psi.pipeline`.

## 7. Acquisition des données et logique de sélection

## 7.1 Endpoints réellement utilisés

### Pipeline built-up

- STAC Planetary Computer : `https://planetarycomputer.microsoft.com/api/stac/v1`

### Pipeline PSI

- STAC CDSE : `https://stac.dataspace.copernicus.eu/v1`
- OData CDSE : `https://catalogue.dataspace.copernicus.eu/odata/v1`
- S3 CDSE principal : `https://eodata.dataspace.copernicus.eu`
- S3 CDSE fallback : `https://eodata.ams.dataspace.copernicus.eu`
- Auth token endpoint : `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`

## 7.2 Gestion des credentials

### Built-up

La branche built-up s’appuie sur Planetary Computer et ne contient pas de logique d’authentification CDSE équivalente à la branche PSI.

### PSI

Le code supporte plusieurs stratégies :

- token d’accès déjà présent ;
- refresh token ;
- login/mot de passe ;
- credentials S3 directs.

Les secrets ne sont pas écrits dans les fichiers de run. Le code met à jour les tokens en mémoire de processus si un refresh est effectué.

## 7.3 Gestion des téléchargements lourds et des erreurs réseau

Le comportement réel côté PSI inclut :

- retries bas niveau botocore ;
- retries applicatifs multi-endpoints ;
- backoff avec jitter ;
- reprise par `Range` sur les membres S3 ;
- préservation des bytes déjà téléchargés ;
- assemblage final seulement après validation.

Cette logique est précisément là pour survivre aux instabilités réseau constatées sur de gros fichiers SLC.

## 8. SNAP dans ce dépôt

## 8.1 Rôle exact de SNAP

SNAP est utilisé pour construire une pile SLC exploitable par StaMPS. Dans ce dépôt, SNAP n’est pas un simple convertisseur ; il assure la préparation radar complète de la pile.

## 8.2 Graphes XML réellement présents

### `resources/snap_graphs/prepare_slc_stack.xml`

Chaîne réelle :

```text
Read
  -> Apply-Orbit-File
  -> TOPSAR-Split
  -> Subset (AOI)
  -> TOPSAR-Deburst
  -> Write (BEAM-DIMAP)
```

Cette étape prépare individuellement le maître et chaque secondaire, swath par swath.

### `resources/snap_graphs/coregister_stack.xml`

Chaîne réelle :

```text
Read master + secondary
  -> Back-Geocoding
  -> Enhanced-Spectral-Diversity
  -> Interferogram
  -> Deburst
  -> TopoPhaseRemoval
  -> Subset
  -> Write coregistered + interferogram
```

### `resources/snap_graphs/stamps_export.xml`

Chaîne réelle :

```text
ProductSet-Reader
  -> TOPSAR-Merge
  -> TopoPhaseRemoval
  -> Subset
  -> StampsExport (psiFormat=true)
```

## 8.3 Répertoires SNAP produits

Par pile :

- `snap/<stack>/prepared/`
- `snap/<stack>/coreg/`
- `snap/<stack>/interferograms/`
- `snap/<stack>/stamps_export/`

Des fichiers texte de suivi sont aussi écrits :

- `coreg_products.txt`
- `ifg_products.txt`

## 9. StaMPS dans ce dépôt

## 9.1 Rôle exact de StaMPS

StaMPS prend l’export SNAP et exécute la chaîne PSI.

Le runner StaMPS vérifie :

- le chemin d’installation ;
- la présence de `bin/mt_prep_snap` ;
- la présence du dossier `matlab/` ;
- la disponibilité de MATLAB ou d’Octave ;
- la présence du script d’export MATLAB.

## 9.2 Commandes réellement lancées

### Préparation

Le code appelle `mt_prep_snap` avec :

```text
mt_prep_snap <masterdate> <datadir> <amp_disp_thresh> <range_patches> <azimuth_patches> <range_overlap> <azimuth_overlap> [mask_file]
```

### Calcul StaMPS

Ensuite le runner lance :

- soit MATLAB : `matlab -batch "..."`
- soit Octave : `octave --quiet --eval "..."`

La commande exécutée appelle :

```text
addpath(...);
cd(...);
stamps(1,8);
run(export_script);
```

## 9.3 Sorties attendues

Le code attend au minimum :

- `stamps/<stack>/export/ps_points.csv`

et crée si besoin un `ps_timeseries.csv` vide avec uniquement l’en-tête.

## 9.4 Export MATLAB réellement fourni

Le script `resources/stamps/export_ps_points.m` exporte :

- `point_id`
- `x_local_m`
- `y_local_m`
- `lon`
- `lat`
- `temporal_coherence`
- `scene_elevation_m`
- `dem_error_phase_per_m`
- `mean_velocity_mm_yr`
- `pre_stability_fraction`
- `post_stability_fraction`
- `first_stable_epoch`
- `residual_height_m`
- `master_day`
- `n_ifg`
- `n_image`

Point critique visible dans le script : `pre_stability_fraction`, `post_stability_fraction` et `residual_height_m` sont actuellement remplis en **NaN par défaut** en attendant une dérivation spécifique au projet à partir des sorties StaMPS réelles.

## 10. Sorties, artefacts et structure des runs

## 10.1 Structure des runs built-up

Le `RunContext` built-up crée :

```text
runs/<group-id>/attempt-001/
  manifests/
  rasters/
  staging/
  vectors/
  reports/
  logs/
```

`group-id` est construit à partir du nom de projet et d’un hash stable de la configuration.

### Artefacts built-up typiques

- `manifests/*.json`
- `rasters/*_s1_*.tif`
- `rasters/*_s2_*.tif`
- `rasters/*_fusion_decision.tif`
- `rasters/cumulative_refined.tif`
- `rasters/zone_mask.tif`
- `vectors/zone_polygons.parquet`
- `vectors/builtup_zones.parquet`
- `vectors/builtup_zones.gpkg`
- `reports/grid.json`
- `reports/resolved_config.yaml`
- `reports/run_report.json`
- `logs/pipeline.log`

## 10.2 Structure des runs PSI

Le `RunContext` PSI crée :

```text
runs_psi/<group-id>/attempt-001/
  manifests/
  raw/
    slc/
    dem/
  snap/
  stamps/
  staging/
  rasters/
  points/
  vectors/
  reports/
  logs/
```

### Artefacts PSI typiques

- `manifests/<stack>.json`
- `raw/slc/<stack>/<scene>.zip`
- `raw/slc/<stack>/<scene>.zip.parts/` pendant le téléchargement
- `snap/<stack>/...`
- `stamps/<stack>/...`
- `points/<stack>_ps_raw.parquet`
- `points/<stack>_ps_emergent.parquet`
- `vectors/psi_candidates.parquet` et/ou `.gpkg`
- `vectors/new_building_candidates.parquet` et/ou `.gpkg`
- `reports/resolved_config.yaml`
- `reports/run_report.json`
- `reports/psi_summary.json`
- `logs/psi_pipeline.log`

Point à noter : `raw/dem/`, `staging/` et `rasters/` sont bien provisionnés par le `RunContext` PSI, mais ils ne sont pas aujourd’hui centraux dans l’orchestrateur PSI courant.

## 10.3 Lecture des sorties PSI

### Parsing Python

`casablanca_psi.psi_results.load_ps_points` accepte :

- soit `lon` et `lat` ;
- soit `x` et `y` projetés.

Si l’export StaMPS ne contient que `x_local_m` et `y_local_m`, Python refuse explicitement le parsing.

### Détection des points émergents

`detect_emergent_ps()` exige les champs :

- `temporal_coherence`
- `pre_stability_fraction`
- `post_stability_fraction`

Il calcule ensuite :

- `stability_gain`
- éventuellement `height_support`
- le booléen `psi_emergent`

## 11. Gestion des erreurs, reprise et robustesse

## 11.1 Rapports de run

Les deux pipelines utilisent la même logique de base pour les rapports :

- `base_run_report`
- `mark_stage`
- `finalize_run_report`
- `mark_running_stages`

Chaque stage possède :

- `status`
- `started_at_utc`
- `completed_at_utc`
- `duration_s`

En cas d’exception, le rapport stocke aussi :

- `error.type`
- `error.message`

## 11.2 Interruption contrôlée

Les deux pipelines utilisent `interruption_guard()` :

- `SIGINT` et `SIGTERM` sont interceptés ;
- une `PipelineInterruptedError` est levée ;
- les stages en cours sont marqués `interrupted` dans le run report.

## 11.3 Robustesse des écritures

Le module `casablanca_builtup.io` écrit rasters et vecteurs dans un fichier temporaire, puis effectue :

1. une validation minimale de l’artefact ;
2. un `replace()` atomique vers le chemin final.

Cela vaut pour :

- les GeoTIFF ;
- les Parquet ;
- les GPKG ;
- les GeoJSON.

## 11.4 Reprise built-up

Le module `casablanca_builtup.resume` :

- recharge les composites s’ils sont valides ;
- recharge les rasters de détection et de raffinage ;
- recharge les rasters de post-traitement ;
- recharge le vecteur `zone_polygons.parquet`.

Les artefacts invalides ou incomplets peuvent être nettoyés puis recalculés.

## 11.5 Reprise PSI

### Manifests

Si `reuse_manifests` est actif, les manifests sont relus.

### Téléchargements

Si `reuse_downloads` est actif :

- les ZIP SLC complets sont réutilisés ;
- les `.zip.part` incomplets sont nettoyés ;
- les `.zip.parts/.../*.part` de niveau membre sont conservés et repris par `Range`.

### SNAP

Si `reuse_snap_outputs` est actif, le runner SNAP réutilise l’export StaMPS si la structure attendue est déjà présente.

### StaMPS

Le champ `reuse_stamps_outputs` existe dans la configuration PSI, mais **aucune logique de réutilisation explicite n’est implémentée dans `casablanca_psi.stamps.StaMPSRunner`**.

## 12. Commandes réelles d’exécution

## 12.1 Pipeline built-up

### Exécution complète

```bash
casablanca-builtup run-pipeline --config configs/casablanca_city.yaml
```

### Exécution stage par stage

```bash
casablanca-builtup acquire-data --config configs/casablanca_city.yaml
casablanca-builtup build-composites --config configs/casablanca_city.yaml
casablanca-builtup detect-s1 --config configs/casablanca_city.yaml
casablanca-builtup refine-s2 --config configs/casablanca_city.yaml
casablanca-builtup polygonize --config configs/casablanca_city.yaml
casablanca-builtup export --config configs/casablanca_city.yaml
```

### Reprise du dernier run

```bash
casablanca-builtup run-pipeline --config configs/casablanca_city.yaml --resume-latest
```

### Évaluation

```bash
casablanca-builtup evaluate-run \
  --run-dir runs/<group>/attempt-001 \
  --reference-raster data/reference/notebook_cumulative_refined.tif
```

## 12.2 Pipeline PSI

### Validation du provider

```bash
casablanca-psi validate-provider --config configs/psi_casablanca_slc.yaml
```

### Exécution complète

```bash
casablanca-psi run-pipeline --config configs/psi_casablanca_slc.yaml
```

### Exécution stage par stage

```bash
casablanca-psi acquire --config configs/psi_casablanca_slc.yaml
casablanca-psi download-slc --config configs/psi_casablanca_slc.yaml
casablanca-psi run-snap --config configs/psi_casablanca_slc.yaml
casablanca-psi run-stamps --config configs/psi_casablanca_slc.yaml
casablanca-psi parse-psi --config configs/psi_casablanca_slc.yaml
casablanca-psi fuse --config configs/psi_casablanca_slc.yaml
```

### Variante minimale

```bash
casablanca-psi run-pipeline --config configs/psi_casablanca_slc_minimal.yaml
```

### Évaluation PSI

```bash
casablanca-psi evaluate \
  --points runs_psi/<group>/attempt-001/vectors/psi_candidates.parquet \
  --reference-points data/reference/psi_points.gpkg \
  --output-dir runs_psi/<group>/attempt-001/reports/evaluation
```

## 12.3 Ordre conseillé d’exécution PSI

Ordre recommandé en exploitation :

1. `validate-provider`
2. `acquire`
3. `download-slc`
4. `run-snap`
5. `run-stamps`
6. `parse-psi`
7. `run-pipeline` si l’on veut pousser jusqu’à l’export final, ou un rerun jusqu’à l’export

La raison est simple : chaque commande stoppe la chaîne à une frontière de stage, mais s’appuie sur les artefacts déjà produits.

## 13. Pipeline minimal vs pipeline complet

Le dépôt contient explicitement **deux modes PSI**.

## 13.1 Mode complet

Fichier : `configs/psi_casablanca_slc.yaml`

Caractéristiques :

- fenêtre temporelle longue ;
- deux piles, ascendante et descendante ;
- `min_scenes: 24` par pile.

Usage :

- production ou expérimentation plus complète ;
- couverture temporelle plus riche ;
- coût de téléchargement élevé.

## 13.2 Mode minimal

Fichier : `configs/psi_casablanca_slc_minimal.yaml`

Caractéristiques :

- une seule pile `asc_rel147_vv` ;
- `min_scenes: 5` ;
- `scene_limit: 5`.

Usage :

- vérification du chaînage externe ;
- validation d’environnement SNAP/StaMPS ;
- tests opérationnels sur le plus petit stack conforme à la contrainte `1 master + 4 secondaires`.

Le dépôt ne contient pas d’équivalent aussi explicite pour un “mode minimal” de la chaîne built-up.

## 14. Pré-requis opérationnels et conseils d’exploitation

## 14.1 Pré-requis logiciels

### Commun

- Python 3.11 ou 3.12 compatible avec le projet ;
- dépendances Python installées depuis `pyproject.toml`.

### Built-up

- accès au Planetary Computer ;
- environnement Python géospatial complet ;
- capacité mémoire suffisante pour les composites.

### PSI

- SNAP installé, avec `gpt` accessible ;
- DEM local disponible ;
- StaMPS installé ;
- MATLAB ou Octave installé ;
- credentials CDSE valides ;
- débit réseau et espace disque suffisants pour les ZIP SLC.

## 14.2 Pré-requis de données

- AOI cohérente avec Casablanca ;
- DEM géoréférencé ;
- pour PSI, acquisitions Sentinel-1 SLC compatibles en direction et relative orbit.

## 14.3 Points de vigilance

- une AOI plus petite n’implique pas un ZIP SLC plus petit : le pipeline télécharge le **produit SLC complet** puis sous-découpe dans SNAP ;
- le coût principal du pipeline PSI est le téléchargement SLC et le prétraitement externe ;
- les runs sont pensés pour être réutilisables via `run_dir` ou `resume_latest` ;
- le hash de configuration change le groupe de run ; changer la configuration peut donc créer un nouvel espace de sortie.

## 14.4 Conseils pratiques

- utiliser le mode PSI minimal pour valider l’environnement avant de viser le mode complet ;
- conserver les répertoires `runs_psi/.../raw/slc/` quand ils contiennent déjà des ZIP complets ;
- surveiller `reports/run_report.json` et le log associé pendant les longs téléchargements ;
- ne pas supposer que la branche PSI exploite déjà l’optique ou l’amplitude simplement parce que les blocs de configuration existent.

## 15. Limites visibles dans le code

Cette section ne décrit que des limites observables dans le dépôt inspecté.

### 15.1 Côté PSI, la fusion réellement orchestrée est PSI-only

Le module `casablanca_psi.fusion` supporte des apports :

- amplitude ;
- optique ;
- contexte.

Mais `casablanca_psi.pipeline` appelle aujourd’hui :

```python
fuse_evidence(psi_points, config.fusion, cluster_buffer_m=config.psi.cluster_buffer_m)
```

Il ne fournit donc pas :

- `amplitude_support`
- `optical_support`
- `context_support`

Conséquence : la fusion réellement exécutée aujourd’hui part uniquement des points PSI. Le score de confiance repose alors sur `psi_primary_weight`, et les autres supports restent à `False`.

### 15.2 Les modules PSI amplitude et S2 existent, mais ne sont pas branchés

Les fichiers suivants existent :

- `src/casablanca_psi/amplitude.py`
- `src/casablanca_psi/s2_refinement.py`

Mais ils ne sont pas invoqués par l’orchestrateur PSI actuel.

### 15.3 Le script d’export StaMPS est volontairement incomplet sur l’émergence

`resources/stamps/export_ps_points.m` écrit bien `ps_points.csv`, mais laisse :

- `pre_stability_fraction`
- `post_stability_fraction`
- `residual_height_m`

à `NaN` tant qu’une dérivation spécifique n’est pas implémentée.

Conséquence : la chaîne Python de détection des points émergents est présente, mais sa validité scientifique dépend de vraies sorties StaMPS enrichies, pas seulement du scaffold actuel.

### 15.4 Le cache StaMPS n’est pas pleinement exploité

`reuse_stamps_outputs` est défini dans la configuration PSI, mais n’est pas utilisé explicitement dans `StaMPSRunner`.

### 15.5 Certains champs de configuration PSI sont aujourd’hui peu ou pas consommés

C’est notamment le cas de :

- `acquisition.provider`
- `acquisition.download_workers`
- `acquisition.s2_catalog_url`
- `acquisition.s2_collection`
- `acquisition.s2_time_window`
- `context.*`
- `snap.workers`

Leur présence documente l’intention architecturale, mais pas nécessairement un comportement déjà actif dans l’orchestrateur courant.

---

## Résumé opérationnel

Le dépôt `casablanca-builtup-pipeline` n’est pas un simple script de détection. C’est un dépôt d’**orchestration géospatiale multi-chaînes** avec :

- une chaîne **built-up** déjà cohérente et exploitable pour produire des zones bâties probables à partir de Sentinel-1 RTC et Sentinel-2 ;
- une chaîne **PSI** structurée autour de Sentinel-1 SLC, SNAP et StaMPS, avec une acquisition robuste, une logique de reprise avancée sur les gros téléchargements S3, un parsing Python et une fusion finale, mais aussi des limites explicites sur l’exploitation scientifique de certains champs StaMPS et sur le branchement des supports amplitude/optique/contexte.

La lecture correcte du dépôt consiste donc à distinguer :

- ce qui est **pleinement orchestré aujourd’hui** ;
- ce qui est **prévu dans l’architecture mais pas encore injecté dans le flux principal** ;
- et ce qui relève d’un **scaffold technique prêt à être spécialisé**.
