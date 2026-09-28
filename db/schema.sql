-- Laplace storage prototype schema.
-- entity: pure content. id = BLAKE3-128 of content; coord = real 4D coordinate (PostGIS POINT ZM,
--   with M carrying the 4th coordinate W in this prototype); tier = level at which it was built
--   (an observation-derived convenience in the prototype); hilbert = 4D Hilbert value of coord.
-- physicality: one per entity, by foreign key. Its geometry is the path: every vertex's X, Y, Z
--   mantissas hold a constituent entity's ID; M holds the run length. Atoms: their own ID point.
-- source: what was ingested, where from; the only place format/type is stored.
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS laplace_proto;
CREATE UNLOGGED TABLE entity (id bytea NOT NULL, coord geometry(PointZM) NOT NULL, tier smallint NOT NULL, hilbert bigint NOT NULL);
CREATE UNLOGGED TABLE physicality (entity bytea NOT NULL, path geometry NOT NULL);
CREATE UNLOGGED TABLE source (trunk bytea NOT NULL, origin text NOT NULL, modality text NOT NULL, format text NOT NULL, bytes bigint NOT NULL, content_blake3 bytea NOT NULL);
CREATE UNLOGGED TABLE entity_stats (id bytea NOT NULL, parents bigint NOT NULL, occurrences numeric NOT NULL);
