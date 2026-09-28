CREATE FUNCTION laplace_vertex_ids(bytea) RETURNS bytea[]
  AS 'MODULE_PATHNAME', 'laplace_vertex_ids' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
