MODULE_big = laplace_proto
OBJS = laplace_proto.o
EXTENSION = laplace_proto
DATA = laplace_proto--0.1.sql
PG_CPPFLAGS = -I/repos/src/blake3/c
SHLIB_LINK = /repos/src/Laplace-Prototype/lib/pic/libblake3_pic.a
PG_CONFIG = /usr/lib/postgresql/18/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
