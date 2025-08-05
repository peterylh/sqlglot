from __future__ import annotations

from sqlglot import exp
from sqlglot.dialects.dialect import (
    approx_count_distinct_sql,
    build_timestamp_trunc,
    property_sql,
    rename_func,
    time_format,
    unit_to_str,
)
from sqlglot.dialects.mysql import MySQL
from sqlglot.tokens import TokenType


def _lag_lead_sql(self, expression: exp.Lag | exp.Lead) -> str:
    return self.func(
        "LAG" if isinstance(expression, exp.Lag) else "LEAD",
        expression.this,
        expression.args.get("offset") or exp.Literal.number(1),
        expression.args.get("default") or exp.null(),
    )


class Doris(MySQL):
    DATE_FORMAT = "'yyyy-MM-dd'"
    DATEINT_FORMAT = "'yyyyMMdd'"
    TIME_FORMAT = "'yyyy-MM-dd HH:mm:ss'"

    class Parser(MySQL.Parser):
        FUNCTIONS = {
            **MySQL.Parser.FUNCTIONS,
            "COLLECT_SET": exp.ArrayUniqueAgg.from_arg_list,
            "DATE_TRUNC": build_timestamp_trunc,
            "MONTHS_ADD": exp.AddMonths.from_arg_list,
            "REGEXP": exp.RegexpLike.from_arg_list,
            "TO_DATE": exp.TsOrDsToDate.from_arg_list,
        }

        FUNCTION_PARSERS = MySQL.Parser.FUNCTION_PARSERS.copy()
        FUNCTION_PARSERS.pop("GROUP_CONCAT")

        PROPERTY_PARSERS = {
            **MySQL.Parser.PROPERTY_PARSERS,
            "PROPERTIES": lambda self: self._parse_wrapped_properties(),
            "UNIQUE": lambda self: self._parse_composite_key_property(exp.UniqueKeyProperty),
            "PARTITION BY": lambda self: self._parse_partition_by_opt_range(),
        }

        def _parse_partitioning_granularity_dynamic(self) -> exp.PartitionByRangePropertyDynamic:
            self._match_text_seq("FROM")
            start = self._parse_wrapped(self._parse_string)
            self._match_text_seq("TO")
            end = self._parse_wrapped(self._parse_string)
            self._match_text_seq("INTERVAL")
            number = self._parse_number()
            unit = self._parse_var(any_token=True)
            every = self.expression(exp.Interval, this=number, unit=unit)
            return self.expression(
                exp.PartitionByRangePropertyDynamic, start=start, end=end, every=every
            )

        def _parse_partition_definition(self) -> exp.Partition:
            self._match_text_seq("PARTITION")

            name = self._parse_id_var()
            self._match_text_seq("VALUES")

            if self._match_text_seq("LESS", "THAN"):
                values = self._parse_wrapped_csv(self._parse_expression)
                if len(values) == 1 and values[0].name.upper() == "MAXVALUE":
                    values = [exp.var("MAXVALUE")]

                part_range = self.expression(exp.PartitionRange, this=name, expressions=values)
                return self.expression(exp.Partition, expressions=[part_range])

            self._match(TokenType.L_BRACKET)
            values = self._parse_csv(lambda: self._parse_wrapped_csv(self._parse_expression))

            self._match(TokenType.R_BRACKET)
            self._match(TokenType.R_PAREN)

            part_range = self.expression(exp.PartitionRange, this=name, expressions=values)
            return self.expression(exp.Partition, expressions=[part_range])

        def _parse_partition_by_opt_range(
            self,
        ) -> exp.PartitionedByProperty | exp.PartitionByRangeProperty:
            if not self._match_text_seq("RANGE"):
                return super()._parse_partitioned_by()

            partition_expressions = self._parse_wrapped_id_vars()
            self._match_l_paren()

            if self._match_text_seq("FROM", advance=False):
                create_expressions = self._parse_csv(self._parse_partitioning_granularity_dynamic)
            elif self._match_text_seq("PARTITION", advance=False):
                create_expressions = self._parse_csv(self._parse_partition_definition)
            else:
                create_expressions = None

            self._match_r_paren()

            return self.expression(
                exp.PartitionByRangeProperty,
                partition_expressions=partition_expressions,
                create_expressions=create_expressions,
            )

    class Generator(MySQL.Generator):
        LAST_DAY_SUPPORTS_DATE_PART = False
        VARCHAR_REQUIRES_SIZE = False
        WITH_PROPERTIES_PREFIX = "PROPERTIES"

        TYPE_MAPPING = {
            **MySQL.Generator.TYPE_MAPPING,
            exp.DataType.Type.TEXT: "STRING",
            exp.DataType.Type.TIMESTAMP: "DATETIME",
            exp.DataType.Type.TIMESTAMPTZ: "DATETIME",
        }

        PROPERTIES_LOCATION = {
            **MySQL.Generator.PROPERTIES_LOCATION,
            exp.UniqueKeyProperty: exp.Properties.Location.POST_SCHEMA,
            exp.PartitionByRangeProperty: exp.Properties.Location.POST_SCHEMA,
        }

        CAST_MAPPING = {}
        TIMESTAMP_FUNC_TYPES = set()

        TRANSFORMS = {
            **MySQL.Generator.TRANSFORMS,
            exp.AddMonths: rename_func("MONTHS_ADD"),
            exp.ApproxDistinct: approx_count_distinct_sql,
            exp.ArgMax: rename_func("MAX_BY"),
            exp.ArgMin: rename_func("MIN_BY"),
            exp.ArrayAgg: rename_func("COLLECT_LIST"),
            exp.ArrayToString: rename_func("ARRAY_JOIN"),
            exp.ArrayUniqueAgg: rename_func("COLLECT_SET"),
            exp.CurrentTimestamp: lambda self, _: self.func("NOW"),
            exp.DateTrunc: lambda self, e: self.func("DATE_TRUNC", e.this, unit_to_str(e)),
            exp.GroupConcat: lambda self, e: self.func(
                "GROUP_CONCAT", e.this, e.args.get("separator") or exp.Literal.string(",")
            ),
            exp.JSONExtractScalar: lambda self, e: self.func("JSON_EXTRACT", e.this, e.expression),
            exp.Lag: _lag_lead_sql,
            exp.Lead: _lag_lead_sql,
            exp.Map: rename_func("ARRAY_MAP"),
            exp.Property: property_sql,
            exp.RegexpLike: rename_func("REGEXP"),
            exp.RegexpSplit: rename_func("SPLIT_BY_STRING"),
            exp.SchemaCommentProperty: lambda self, e: self.naked_property(e),
            exp.Split: rename_func("SPLIT_BY_STRING"),
            exp.StringToArray: rename_func("SPLIT_BY_STRING"),
            exp.StrToUnix: lambda self, e: self.func("UNIX_TIMESTAMP", e.this, self.format_time(e)),
            exp.TimeStrToDate: rename_func("TO_DATE"),
            exp.TsOrDsAdd: lambda self, e: self.func("DATE_ADD", e.this, e.expression),
            exp.TsOrDsToDate: lambda self, e: self.func("TO_DATE", e.this),
            exp.TimeToUnix: rename_func("UNIX_TIMESTAMP"),
            exp.TimestampTrunc: lambda self, e: self.func("DATE_TRUNC", e.this, unit_to_str(e)),
            exp.UnixToStr: lambda self, e: self.func(
                "FROM_UNIXTIME", e.this, time_format("doris")(self, e)
            ),
            exp.UnixToTime: rename_func("FROM_UNIXTIME"),
        }

        # https://github.com/apache/doris/blob/e4f41dbf1ec03f5937fdeba2ee1454a20254015b/fe/fe-core/src/main/antlr4/org/apache/doris/nereids/DorisLexer.g4#L93
        RESERVED_KEYWORDS = {
            "account_lock",
            "account_unlock",
            "add",
            "adddate",
            "admin",
            "after",
            "agg_state",
            "aggregate",
            "alias",
            "all",
            "alter",
            "analyze",
            "analyzed",
            "and",
            "anti",
            "append",
            "array",
            "array_range",
            "as",
            "asc",
            "at",
            "authors",
            "auto",
            "auto_increment",
            "backend",
            "backends",
            "backup",
            "begin",
            "belong",
            "between",
            "bigint",
            "bin",
            "binary",
            "binlog",
            "bitand",
            "bitmap",
            "bitmap_union",
            "bitor",
            "bitxor",
            "blob",
            "boolean",
            "brief",
            "broker",
            "buckets",
            "build",
            "builtin",
            "bulk",
            "by",
            "cached",
            "call",
            "cancel",
            "case",
            "cast",
            "catalog",
            "catalogs",
            "chain",
            "char",
            "character",
            "charset",
            "check",
            "clean",
            "cluster",
            "clusters",
            "collate",
            "collation",
            "collect",
            "column",
            "columns",
            "comment",
            "commit",
            "committed",
            "compact",
            "complete",
            "config",
            "connection",
            "connection_id",
            "consistent",
            "constraint",
            "constraints",
            "convert",
            "copy",
            "count",
            "create",
            "creation",
            "cron",
            "cross",
            "cube",
            "current",
            "current_catalog",
            "current_date",
            "current_time",
            "current_timestamp",
            "current_user",
            "data",
            "database",
            "databases",
            "date",
            "date_add",
            "date_ceil",
            "date_diff",
            "date_floor",
            "date_sub",
            "dateadd",
            "datediff",
            "datetime",
            "datetimev2",
            "datev2",
            "datetimev1",
            "datev1",
            "day",
            "days_add",
            "days_sub",
            "decimal",
            "decimalv2",
            "decimalv3",
            "decommission",
            "default",
            "deferred",
            "delete",
            "demand",
            "desc",
            "describe",
            "diagnose",
            "disk",
            "distinct",
            "distinctpc",
            "distinctpcsa",
            "distributed",
            "distribution",
            "div",
            "do",
            "doris_internal_table_id",
            "double",
            "drop",
            "dropp",
            "dual",
            "duplicate",
            "dynamic",
            "else",
            "enable",
            "encryptkey",
            "encryptkeys",
            "end",
            "ends",
            "engine",
            "engines",
            "enter",
            "errors",
            "events",
            "every",
            "except",
            "exclude",
            "execute",
            "exists",
            "expired",
            "explain",
            "export",
            "extended",
            "external",
            "extract",
            "failed_login_attempts",
            "false",
            "fast",
            "feature",
            "fields",
            "file",
            "filter",
            "first",
            "float",
            "follower",
            "following",
            "for",
            "foreign",
            "force",
            "format",
            "free",
            "from",
            "frontend",
            "frontends",
            "full",
            "function",
            "functions",
            "generic",
            "global",
            "grant",
            "grants",
            "graph",
            "group",
            "grouping",
            "groups",
            "hash",
            "having",
            "hdfs",
            "help",
            "histogram",
            "hll",
            "hll_union",
            "hostname",
            "hour",
            "hub",
            "identified",
            "if",
            "ignore",
            "immediate",
            "in",
            "incremental",
            "index",
            "indexes",
            "infile",
            "inner",
            "insert",
            "install",
            "int",
            "integer",
            "intermediate",
            "intersect",
            "interval",
            "into",
            "inverted",
            "ipv4",
            "ipv6",
            "is",
            "is_not_null_pred",
            "is_null_pred",
            "isnull",
            "isolation",
            "job",
            "jobs",
            "join",
            "json",
            "jsonb",
            "key",
            "keys",
            "kill",
            "label",
            "largeint",
            "last",
            "lateral",
            "ldap",
            "ldap_admin_password",
            "left",
            "less",
            "level",
            "like",
            "limit",
            "lines",
            "link",
            "list",
            "load",
            "local",
            "localtime",
            "localtimestamp",
            "location",
            "lock",
            "logical",
            "low_priority",
            "manual",
            "map",
            "match",
            "match_all",
            "match_any",
            "match_phrase",
            "match_phrase_edge",
            "match_phrase_prefix",
            "match_regexp",
            "materialized",
            "max",
            "maxvalue",
            "memo",
            "merge",
            "migrate",
            "migrations",
            "min",
            "minus",
            "minute",
            "modify",
            "month",
            "mtmv",
            "name",
            "names",
            "natural",
            "negative",
            "never",
            "next",
            "ngram_bf",
            "no",
            "non_nullable",
            "not",
            "null",
            "nulls",
            "observer",
            "of",
            "offset",
            "on",
            "only",
            "open",
            "optimized",
            "or",
            "order",
            "outer",
            "outfile",
            "over",
            "overwrite",
            "parameter",
            "parsed",
            "partition",
            "partitions",
            "password",
            "password_expire",
            "password_history",
            "password_lock_time",
            "password_reuse",
            "path",
            "pause",
            "percent",
            "period",
            "permissive",
            "physical",
            "plan",
            "process",
            "plugin",
            "plugins",
            "policy",
            "preceding",
            "prepare",
            "primary",
            "proc",
            "procedure",
            "processlist",
            "profile",
            "properties",
            "property",
            "quantile_state",
            "quantile_union",
            "query",
            "quota",
            "random",
            "range",
            "read",
            "real",
            "rebalance",
            "recover",
            "recycle",
            "refresh",
            "references",
            "regexp",
            "release",
            "rename",
            "repair",
            "repeatable",
            "replace",
            "replace_if_not_null",
            "replica",
            "repositories",
            "repository",
            "resource",
            "resources",
            "restore",
            "restrictive",
            "resume",
            "returns",
            "revoke",
            "rewritten",
            "right",
            "rlike",
            "role",
            "roles",
            "rollback",
            "rollup",
            "routine",
            "row",
            "rows",
            "s3",
            "sample",
            "schedule",
            "scheduler",
            "schema",
            "schemas",
            "second",
            "select",
            "semi",
            "sequence",
            "serializable",
            "session",
            "set",
            "sets",
            "shape",
            "show",
            "signed",
            "skew",
            "smallint",
            "snapshot",
            "soname",
            "split",
            "sql_block_rule",
            "start",
            "starts",
            "stats",
            "status",
            "stop",
            "storage",
            "stream",
            "streaming",
            "string",
            "struct",
            "subdate",
            "sum",
            "superuser",
            "switch",
            "sync",
            "system",
            "table",
            "tables",
            "tablesample",
            "tablet",
            "tablets",
            "task",
            "tasks",
            "temporary",
            "terminated",
            "text",
            "than",
            "then",
            "time",
            "timestamp",
            "timestampadd",
            "timestampdiff",
            "tinyint",
            "to",
            "transaction",
            "trash",
            "tree",
            "triggers",
            "trim",
            "true",
            "truncate",
            "type",
            "type_cast",
            "types",
            "unbounded",
            "uncommitted",
            "uninstall",
            "union",
            "unique",
            "unlock",
            "unsigned",
            "update",
            "use",
            "user",
            "using",
            "value",
            "values",
            "varchar",
            "variables",
            "variant",
            "vault",
            "verbose",
            "version",
            "view",
            "warnings",
            "week",
            "when",
            "where",
            "whitelist",
            "with",
            "work",
            "workload",
            "write",
            "xor",
            "year",
        }

        def partition_sql(self, expression: exp.Partition) -> str:
            parent = expression.parent
            if isinstance(parent, exp.PartitionByRangeProperty):
                return ", ".join(self.sql(e) for e in expression.expressions)
            return super().partition_sql(expression)

        def partitionrange_sql(self, expression: exp.PartitionRange) -> str:
            name = self.sql(expression, "this")
            values = expression.expressions

            if len(values) != 1:
                # Multiple values: use VALUES [ ... )
                if values and isinstance(values[0], list):
                    values_sql = ", ".join(
                        f"({', '.join(self.sql(v) for v in inner)})" for inner in values
                    )
                else:
                    values_sql = ", ".join(f"({self.sql(v)})" for v in values)

                return f"PARTITION {name} VALUES [{values_sql})"

            return f"PARTITION {name} VALUES LESS THAN ({self.sql(values[0])})"

        def partitionbyrangepropertydynamic_sql(self, expression):
            # Generates: FROM ("start") TO ("end") INTERVAL N UNIT
            start = self.sql(expression, "start")
            end = self.sql(expression, "end")
            every = expression.args.get("every")

            if every:
                number = self.sql(every, "this")
                interval = f"INTERVAL {number} {self.sql(every, 'unit')}"
            else:
                interval = ""

            return f"FROM ({start}) TO ({end}) {interval}"

        def partitionbyrangeproperty_sql(self, expression):
            partition_expressions = ", ".join(
                self.sql(e) for e in expression.args.get("partition_expressions") or []
            )
            create_expressions = expression.args.get("create_expressions") or []
            # Handle both static and dynamic partition definitions
            create_sql = ", ".join(self.sql(e) for e in create_expressions)
            return f"PARTITION BY RANGE ({partition_expressions}) ({create_sql})"

        def table_sql(self, expression: exp.Table, sep: str = " AS ") -> str:
            """Override table_sql to avoid AS keyword in UPDATE and DELETE statements."""
            # Check if this table is part of an UPDATE or DELETE statement or their clauses
            if expression.find_ancestor(exp.Update, exp.Delete):
                return super().table_sql(expression, sep=" ")
            return super().table_sql(expression, sep=sep)

        def _extract_literal_value(self, expression):
            """Extract literal value from CAST expressions or return the expression as-is.

            Doris doesn't support CAST in partition definitions, so we need to extract
            the literal value from CAST expressions.
            """
            if isinstance(expression, exp.Cast):
                # Extract the literal value from CAST
                return expression.this
            return expression

        def _clean_partition_name(self, expr):
            """Extract and clean partition name from expression.

            This method handles both CAST and Literal expressions and returns
            a cleaned partition name suitable for Doris.
            """
            # First extract the literal value from CAST if needed
            literal_expr = self._extract_literal_value(expr)

            if isinstance(literal_expr, exp.Literal):
                clean_name = literal_expr.this.replace("-", "")
                return f"p{clean_name}"

            # Fallback: use the full expression
            expr_sql = self.sql(expr)
            clean_name = (
                expr_sql.replace("'", "")
                .replace('"', "")
                .replace("-", "")
                .replace(":", "")
                .replace("_", "")
            )
            return f"p{clean_name}"

        def addpartition_sql(self, expression: exp.AddPartition) -> str:
            """Convert Greenplum ADD PARTITION with START/END to Doris VALUES syntax.

            Doris only supports two specific cases:
            1. START INCLUSIVE END EXCLUSIVE -> Doris left-closed, right-open interval [start, end)
            2. END EXCLUSIVE (no START) -> Doris VALUES LESS THAN

            All other combinations will raise an error.
            """
            exists_sql = " IF NOT EXISTS" if expression.args.get("exists") else ""
            partition_name = self.sql(expression, "this")

            # Check if we have START/END range specification
            has_start = expression.args.get("start") is not None
            has_end = expression.args.get("end") is not None
            start_inclusive = expression.args.get("start_inclusive")
            end_inclusive = expression.args.get("end_inclusive")

            if has_start and has_end:
                # Case 1: START INCLUSIVE END EXCLUSIVE -> Doris left-closed, right-open interval
                if start_inclusive is True and end_inclusive is False:
                    # Convert to Doris VALUES [start, end) syntax
                    # Extract literal values from CAST expressions
                    start_expr = self._extract_literal_value(expression.args.get("start"))
                    end_expr = self._extract_literal_value(expression.args.get("end"))
                    start_value = self.sql(start_expr)
                    end_value = self.sql(end_expr)
                    return f"ADD PARTITION{exists_sql} {partition_name} VALUES [({start_value}), ({end_value}))"
                else:
                    # All other START/END combinations are not supported
                    start_type = "INCLUSIVE" if start_inclusive else "EXCLUSIVE"
                    end_type = "INCLUSIVE" if end_inclusive else "EXCLUSIVE"
                    raise ValueError(
                        f"Doris partition conversion error: Unsupported START/END combination. "
                        f"Got START {start_type} END {end_type} in partition '{partition_name}'. "
                        f"Only 'START INCLUSIVE END EXCLUSIVE' is supported for range partitions."
                    )

            elif has_end and not has_start:
                # Case 2: Only END EXCLUSIVE -> Doris VALUES LESS THAN
                if end_inclusive is False:
                    # Extract literal value from CAST expression
                    end_expr = self._extract_literal_value(expression.args.get("end"))
                    end_value = self.sql(end_expr)
                    return (
                        f"ADD PARTITION{exists_sql} {partition_name} VALUES LESS THAN ({end_value})"
                    )
                else:
                    raise ValueError(
                        f"Doris partition conversion error: Unsupported END type. "
                        f"Got END INCLUSIVE in partition '{partition_name}'. "
                        f"Only 'END EXCLUSIVE' is supported when START is not specified."
                    )

            elif has_start and not has_end:
                # Only START specified - not supported in Doris conversion
                start_type = "INCLUSIVE" if start_inclusive else "EXCLUSIVE"
                raise ValueError(
                    f"Doris partition conversion error: Unsupported partition specification. "
                    f"Got START {start_type} without END in partition '{partition_name}'. "
                    f"Doris requires either 'START INCLUSIVE END EXCLUSIVE' or 'END EXCLUSIVE' only."
                )

            else:
                # No range specified, use basic syntax (this should work for simple partitions)
                return f"ADD PARTITION{exists_sql} {partition_name}"

        def droppartition_sql(self, expression: exp.DropPartition) -> str:
            """Generate SQL for DROP PARTITION in Doris."""
            exists_sql = " IF EXISTS" if expression.args.get("exists") else ""

            if expression.args.get("for_clause"):
                # Convert Greenplum FOR syntax to standard partition name
                if expression.expressions:
                    expr = expression.expressions[0]
                    partition_name = self._clean_partition_name(expr)
                    return f"DROP PARTITION{exists_sql} {partition_name}"
                else:
                    return f"DROP PARTITION{exists_sql}"
            else:
                # Standard syntax
                expressions_sql = self.expressions(expression)
                return f"DROP PARTITION{exists_sql} {expressions_sql}"

        def alter_sql(self, expression: exp.Alter) -> str:
            """Override ALTER SQL generation to handle TruncatePartition specially.

            In Doris, TRUNCATE PARTITION should generate a standalone TRUNCATE TABLE statement
            instead of being part of an ALTER TABLE statement.
            """
            actions = expression.args["actions"]

            # Special handling for TruncatePartition - generate standalone TRUNCATE TABLE
            if len(actions) == 1 and isinstance(actions[0], exp.TruncatePartition):
                truncate_action = actions[0]
                table_name = self.sql(expression.this)

                if truncate_action.args.get("for_clause"):
                    # Convert Greenplum FOR syntax to Doris TRUNCATE TABLE syntax
                    expr = truncate_action.args.get("expression")
                    if expr:
                        partition_name = self._clean_partition_name(expr)
                        return f"TRUNCATE TABLE {table_name} PARTITION({partition_name})"
                    else:
                        return f"TRUNCATE TABLE {table_name}"
                else:
                    # Standard syntax with partition names
                    if truncate_action.expressions:
                        expressions_sql = self.expressions(truncate_action, flat=True)
                        return f"TRUNCATE TABLE {table_name} PARTITION({expressions_sql})"
                    else:
                        return f"TRUNCATE TABLE {table_name}"

            # For all other ALTER statements, use the default behavior
            return super().alter_sql(expression)

        def truncatepartition_sql(self, expression: exp.TruncatePartition) -> str:
            """Generate SQL for TRUNCATE PARTITION in Doris.

            This method should not be called directly when TruncatePartition is part of an ALTER statement,
            as alter_sql() handles it specially. This is kept for compatibility.
            """
            # Get the table name from the parent ALTER statement
            parent = expression.parent
            table_name = "unknown_table"  # fallback

            if parent and hasattr(parent, "this") and parent.this:
                table_name = self.sql(parent.this)

            if expression.args.get("for_clause"):
                # Convert Greenplum FOR syntax to Doris TRUNCATE TABLE syntax
                expr = expression.args.get("expression")
                if expr:
                    partition_name = self._clean_partition_name(expr)
                    return f"TRUNCATE TABLE {table_name} PARTITION({partition_name})"
                else:
                    return f"TRUNCATE TABLE {table_name}"
            else:
                # Standard syntax with partition names
                if expression.expressions:
                    expressions_sql = self.expressions(expression, flat=True)
                    return f"TRUNCATE TABLE {table_name} PARTITION({expressions_sql})"
                else:
                    return f"TRUNCATE TABLE {table_name}"
