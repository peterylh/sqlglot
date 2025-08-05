from __future__ import annotations

import typing as t

from sqlglot import exp
from sqlglot.dialects.postgres import Postgres
from sqlglot.tokens import TokenType
from sqlglot.helper import seq_get


class Greenplum(Postgres):
    """
    Greenplum dialect that extends PostgreSQL to support Greenplum-specific partition syntax.
    
    Greenplum uses specific partition syntax like:
    - ALTER TABLE ... ADD PARTITION ... START (...) INCLUSIVE END (...) EXCLUSIVE
    - ALTER TABLE ... TRUNCATE PARTITION FOR (...)
    - ALTER TABLE ... DROP PARTITION IF EXISTS FOR (...)
    """
    
    class Tokenizer(Postgres.Tokenizer):
        KEYWORDS = {
            **Postgres.Tokenizer.KEYWORDS,
            "FOR": TokenType.FOR,
        }
    
    class Parser(Postgres.Parser):
        STATEMENT_PARSERS = {
            **Postgres.Parser.STATEMENT_PARSERS,
            TokenType.ALTER: lambda self: self._parse_alter(),
        }
        
        ALTER_PARSERS = {
            **Postgres.Parser.ALTER_PARSERS,
            "ADD": lambda self: self._parse_alter_table_add(),
            "DROP": lambda self: self._parse_alter_table_drop(),
            "TRUNCATE": lambda self: self._parse_alter_table_truncate(),
        }
        def _parse_alter_table_add(self) -> t.List[exp.Expression]:
            def _parse_add_alteration() -> t.Optional[exp.Expression]:
                # Check for PARTITION (Greenplum-specific)
                if self._match(TokenType.PARTITION):
                    # Check for IF NOT EXISTS after PARTITION
                    exists = None
                    if self._match_text_seq("IF", "NOT", "EXISTS"):
                        exists = True
                    # Parse partition name
                    partition_name = self._parse_field(any_token=True)
                    
                    # Parse START clause
                    start = None
                    start_inclusive = None
                    if self._match_text_seq("START"):
                        start = self._parse_wrapped(self._parse_assignment)
                        if self._match_text_seq("INCLUSIVE"):
                            start_inclusive = True
                        elif self._match_text_seq("EXCLUSIVE"):
                            start_inclusive = False
                        else:
                            start_inclusive = True  # Default to INCLUSIVE if not specified
                    
                    # Parse END clause
                    end = None
                    end_inclusive = None
                    if self._match_text_seq("END"):
                        end = self._parse_wrapped(self._parse_assignment)
                        if self._match_text_seq("EXCLUSIVE"):
                            end_inclusive = False
                        elif self._match_text_seq("INCLUSIVE"):
                            end_inclusive = True
                        else:
                            end_inclusive = True  # Default to INCLUSIVE if not specified
                    
                    # Parse WITH clause for properties
                    properties = None
                    if self._match(TokenType.WITH):
                        properties = self._parse_wrapped_csv(self._parse_property)
                    
                    return self.expression(
                        exp.AddPartition,
                        exists=exists,
                        this=partition_name,
                        start=start,
                        start_inclusive=start_inclusive,
                        end=end,
                        end_inclusive=end_inclusive,
                        properties=properties,
                    )
                
                # Check for constraints
                if self._match_set(self.ADD_CONSTRAINT_TOKENS, advance=False):
                    return self.expression(
                        exp.AddConstraint, expressions=self._parse_csv(self._parse_constraint)
                    )

                # Check for column definitions
                column_def = self._parse_add_column()
                if isinstance(column_def, exp.ColumnDef):
                    return column_def

                return None

            from sqlglot.helper import ensure_list
            if not self._match_set(self.ADD_CONSTRAINT_TOKENS, advance=False) and (
                not self.dialect.ALTER_TABLE_ADD_REQUIRED_FOR_EACH_COLUMN
                or self._match_text_seq("COLUMNS")
            ):
                schema = self._parse_schema()

                return (
                    ensure_list(schema)
                    if schema
                    else self._parse_csv(self._parse_column_def_with_exists)
                )

            return self._parse_csv(_parse_add_alteration)
        
        def _parse_alter_table_drop(self) -> t.List[exp.Expression]:
            index = self._index - 1

            # Check for PARTITION first
            if self._match(TokenType.PARTITION):
                # Parse IF EXISTS after PARTITION
                partition_exists = self._parse_exists()
                return [self._parse_drop_partition_greenplum(exists=partition_exists)]
            
            # Check for IF EXISTS followed by PARTITION
            partition_exists = self._parse_exists()
            if self._match(TokenType.PARTITION):
                return [self._parse_drop_partition_greenplum(exists=partition_exists)]

            self._retreat(index)
            return self._parse_csv(self._parse_drop_column)
        
        def _parse_drop_partition_greenplum(self, exists: t.Optional[bool] = None) -> exp.DropPartition:
            """Parse Greenplum-specific DROP PARTITION syntax."""
            # PARTITION token already matched in _parse_alter_table_drop
            
            # Handle IF EXISTS
            if exists is None:
                exists = self._parse_exists()
            
            # Handle FOR clause
            for_clause = False
            if self._match(TokenType.FOR):
                expressions = [self._parse_wrapped(self._parse_expression)]
                for_clause = True
            else:
                expressions = self._parse_csv(self._parse_partition)
            
            return self.expression(
                exp.DropPartition, expressions=expressions, exists=exists, for_clause=for_clause
            )
        

        
        def _parse_alter(self) -> exp.Alter | exp.Command:
            """Override to handle ALTER TABLE with Greenplum-specific syntax."""
            from sqlglot.helper import ensure_list
            
            start = self._prev

            alter_token = self._match_set(self.ALTERABLES) and self._prev
            if not alter_token:
                return self._parse_as_command(start)

            # Skip parsing exists here for Greenplum to avoid consuming IF NOT EXISTS
            # exists = self._parse_exists()  # This line is commented out
            exists = None
            only = self._match_text_seq("ONLY")
            this = self._parse_table(schema=True)
            cluster = self._parse_on_property() if self._match(TokenType.ON) else None

            if self._next:
                self._advance()

            parser = self.ALTER_PARSERS.get(self._prev.text.upper()) if self._prev else None
            if parser:
                actions = ensure_list(parser(self))
                not_valid = self._match_text_seq("NOT", "VALID")
                options = self._parse_csv(self._parse_property)

                if not self._curr and actions:
                    return self.expression(
                        exp.Alter,
                        this=this,
                        kind=alter_token.text.upper(),
                        exists=exists,
                        actions=actions,
                        only=only,
                        options=options,
                        cluster=cluster,
                        not_valid=not_valid,
                    )

            return self._parse_as_command(start)
        
        def _parse_alter_table_truncate(self) -> t.Optional[exp.Expression]:
            # Handle TRUNCATE PARTITION syntax
            if self._match(TokenType.PARTITION):
                if self._match(TokenType.FOR):
                    # TRUNCATE PARTITION FOR (values)
                    values = self._parse_wrapped_csv(self._parse_bitwise)
                    return self.expression(exp.TruncatePartition, expression=values[0] if values else None, for_clause=True)
                else:
                    # TRUNCATE PARTITION partition_name
                    partition_name = self._parse_id_var()
                    return self.expression(exp.TruncatePartition, expressions=[partition_name])
            return None
    
    class Generator(Postgres.Generator):
        def addpartition_sql(self, expression: exp.AddPartition) -> str:
            """Generate SQL for Greenplum ADD PARTITION."""
            exists_sql = " IF NOT EXISTS" if expression.args.get("exists") else ""
            partition_name = self.sql(expression, "this")
            
            sql = f"ADD PARTITION{exists_sql} {partition_name}"
            
            # Add START clause
            if expression.args.get("start"):
                start_expr = self.sql(expression, "start")
                start_type = " INCLUSIVE" if expression.args.get("start_inclusive") else " EXCLUSIVE"
                sql += f" START ({start_expr}){start_type}"
            
            # Add END clause
            if expression.args.get("end"):
                end_expr = self.sql(expression, "end")
                end_type = " INCLUSIVE" if expression.args.get("end_inclusive") else " EXCLUSIVE"
                sql += f" END ({end_expr}){end_type}"
            
            # Add WITH clause for properties
            if expression.args.get("properties"):
                properties = expression.args.get("properties")
                if properties:
                    properties_sql = self.expressions(properties, flat=True)
                    sql += f" WITH ({properties_sql})"
            
            return sql
        
        def droppartition_sql(self, expression: exp.DropPartition) -> str:
            """Generate SQL for Greenplum DROP PARTITION."""
            exists_sql = " IF EXISTS" if expression.args.get("exists") else ""
            
            if expression.args.get("for_clause"):
                # Greenplum FOR syntax
                expressions_sql = ", ".join(f"({self.sql(e)})" for e in expression.expressions)
                return f"DROP PARTITION{exists_sql} FOR {expressions_sql}"
            else:
                # Standard syntax
                expressions_sql = self.expressions(expression)
                return f"DROP PARTITION{exists_sql} {expressions_sql}"
        
        def truncatepartition_sql(self, expression: exp.TruncatePartition) -> str:
            """Generate SQL for Greenplum TRUNCATE PARTITION."""
            if expression.args.get("for_clause"):
                # Greenplum FOR syntax
                expr_sql = self.sql(expression, "expression")
                return f"TRUNCATE PARTITION FOR ({expr_sql})"
            elif expression.expressions:
                # Standard syntax with partition names
                expressions_sql = self.expressions(expression, flat=True)
                return f"TRUNCATE PARTITION {expressions_sql}"
            else:
                # Fallback
                return "TRUNCATE PARTITION"