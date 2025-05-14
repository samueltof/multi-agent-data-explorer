-- 1) complexes: one row per T-cell clone (α+β pair), with shared fields
CREATE TABLE complexes (
  complex_id       INTEGER PRIMARY KEY,     -- complex.id
  mhc_a            TEXT     NOT NULL,       -- mhc.a
  mhc_b            TEXT     NOT NULL,       -- mhc.b
  mhc_class        TEXT     NOT NULL,       -- mhc.class
  antigen_epitope  TEXT     NOT NULL,       -- antigen.epitope
  antigen_gene     TEXT     NOT NULL,       -- antigen.gene
  antigen_species  TEXT     NOT NULL,       -- antigen.species
  reference_id     TEXT     NOT NULL,       -- reference.id
  method           TEXT     NOT NULL,       -- method (JSON blob)
  meta             TEXT     NOT NULL        -- meta (JSON blob)
);

-- 2) chains: one row per α or β chain within a complex
CREATE TABLE chains (
  chain_id            INTEGER PRIMARY KEY AUTOINCREMENT,
  complex_id          INTEGER     NOT NULL REFERENCES complexes(complex_id),
  gene                TEXT        NOT NULL,  -- gene (“TRA”/“TRB”)
  cdr3                TEXT        NOT NULL,  -- cdr3
  v_segm              TEXT        NOT NULL,  -- v.segm
  j_segm              TEXT        NOT NULL,  -- j.segm
  species             TEXT        NOT NULL,  -- species (TCR parent species)
  cdr3fix             TEXT        NOT NULL,  -- cdr3fix (JSON blob)
  web_method          TEXT        NOT NULL,  -- web.method
  web_method_seq      TEXT        NOT NULL,  -- web.method.seq
  web_cdr3fix_nc      TEXT        NOT NULL,  -- web.cdr3fix.nc
  web_cdr3fix_unmp    TEXT        NOT NULL,  -- web.cdr3fix.unmp
  vdjdb_score         INTEGER     NOT NULL   -- vdjdb.score
);
