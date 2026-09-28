\timing on
ALTER TABLE entity SET LOGGED;
ALTER TABLE physicality SET LOGGED;
ALTER TABLE source SET LOGGED;
ALTER TABLE entity_stats SET LOGGED;
ALTER TABLE entity ADD PRIMARY KEY (id);
ALTER TABLE physicality ADD PRIMARY KEY (entity);
ALTER TABLE physicality ADD FOREIGN KEY (entity) REFERENCES entity (id);
ALTER TABLE entity_stats ADD PRIMARY KEY (id);
CREATE INDEX entity_coord_gist ON entity USING gist (coord gist_geometry_ops_nd);
CREATE INDEX entity_hilbert ON entity (hilbert);
CREATE INDEX physicality_containers ON physicality USING gin (laplace_vertex_ids(st_asewkb(path)));
CREATE INDEX source_trunk ON source (trunk);
ANALYZE;
