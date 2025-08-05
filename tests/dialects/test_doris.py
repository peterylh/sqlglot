from tests.dialects.test_dialect import Validator
import sqlglot


class TestDoris(Validator):
    dialect = "doris"

    def test_doris(self):
        self.validate_all(
            "SELECT TO_DATE('2020-02-02 00:00:00')",
            write={
                "doris": "SELECT TO_DATE('2020-02-02 00:00:00')",
                "oracle": "SELECT CAST('2020-02-02 00:00:00' AS DATE)",
            },
        )
        self.validate_all(
            "SELECT MAX_BY(a, b), MIN_BY(c, d)",
            read={
                "clickhouse": "SELECT argMax(a, b), argMin(c, d)",
            },
        )
        self.validate_all(
            "SELECT ARRAY_SUM(x -> x * x, ARRAY(2, 3))",
            read={
                "clickhouse": "SELECT arraySum(x -> x*x, [2, 3])",
            },
            write={
                "clickhouse": "SELECT arraySum(x -> x * x, [2, 3])",
                "doris": "SELECT ARRAY_SUM(x -> x * x, ARRAY(2, 3))",
            },
        )
        self.validate_all(
            "MONTHS_ADD(d, n)",
            read={
                "oracle": "ADD_MONTHS(d, n)",
            },
            write={
                "doris": "MONTHS_ADD(d, n)",
                "oracle": "ADD_MONTHS(d, n)",
            },
        )
        self.validate_all(
            """SELECT JSON_EXTRACT(CAST('{"key": 1}' AS JSONB), '$.key')""",
            read={
                "postgres": """SELECT '{"key": 1}'::jsonb ->> 'key'""",
            },
            write={
                "doris": """SELECT JSON_EXTRACT(CAST('{"key": 1}' AS JSONB), '$.key')""",
                "postgres": """SELECT JSON_EXTRACT_PATH(CAST('{"key": 1}' AS JSONB), 'key')""",
            },
        )
        self.validate_all(
            "SELECT GROUP_CONCAT('aa', ',')",
            read={
                "doris": "SELECT GROUP_CONCAT('aa', ',')",
                "mysql": "SELECT GROUP_CONCAT('aa' SEPARATOR ',')",
                "postgres": "SELECT STRING_AGG('aa', ',')",
            },
        )
        self.validate_all(
            "SELECT LAG(1, 1, NULL) OVER (ORDER BY 1)",
            read={
                "doris": "SELECT LAG(1, 1, NULL) OVER (ORDER BY 1)",
                "postgres": "SELECT LAG(1) OVER (ORDER BY 1)",
            },
        )
        self.validate_all(
            "SELECT LAG(1, 2, NULL) OVER (ORDER BY 1)",
            read={
                "doris": "SELECT LAG(1, 2, NULL) OVER (ORDER BY 1)",
                "postgres": "SELECT LAG(1, 2) OVER (ORDER BY 1)",
            },
        )
        self.validate_all(
            "SELECT LEAD(1, 1, NULL) OVER (ORDER BY 1)",
            read={
                "doris": "SELECT LEAD(1, 1, NULL) OVER (ORDER BY 1)",
                "postgres": "SELECT LEAD(1) OVER (ORDER BY 1)",
            },
        )
        self.validate_all(
            "SELECT LEAD(1, 2, NULL) OVER (ORDER BY 1)",
            read={
                "doris": "SELECT LEAD(1, 2, NULL) OVER (ORDER BY 1)",
                "postgres": "SELECT LEAD(1, 2) OVER (ORDER BY 1)",
            },
        )
        self.validate_identity("""JSON_TYPE('{"foo": "1" }', '$.foo')""")

    def test_identity(self):
        self.validate_identity("CREATE TABLE t (c INT) PROPERTIES ('x'='y')")
        self.validate_identity("CREATE TABLE t (c INT) COMMENT 'c'")
        self.validate_identity("COALECSE(a, b, c, d)")
        self.validate_identity("SELECT CAST(`a`.`b` AS INT) FROM foo")
        self.validate_identity("SELECT APPROX_COUNT_DISTINCT(a) FROM x")
        self.validate_identity(
            "CREATE TABLE IF NOT EXISTS example_tbl_unique (user_id BIGINT NOT NULL, user_name VARCHAR(50) NOT NULL, city VARCHAR(20), age SMALLINT, sex TINYINT) UNIQUE KEY (user_id, user_name) DISTRIBUTED BY HASH (user_id) BUCKETS 10 PROPERTIES ('enable_unique_key_merge_on_write'='true')"
        )
        self.validate_identity("INSERT OVERWRITE TABLE test PARTITION(p1, p2) VALUES (1, 2)")

    def test_time(self):
        self.validate_identity("TIMESTAMP('2022-01-01')")

    def test_regex(self):
        self.validate_all(
            "SELECT REGEXP_LIKE(abc, '%foo%')",
            write={
                "doris": "SELECT REGEXP(abc, '%foo%')",
            },
        )

    def test_analyze(self):
        self.validate_identity("ANALYZE TABLE tbl")
        self.validate_identity("ANALYZE DATABASE db")
        self.validate_identity("ANALYZE TABLE TBL(c1, c2)")

    def test_key(self):
        self.validate_identity("CREATE TABLE test_table (c1 INT, c2 INT) UNIQUE KEY (c1)")
        self.validate_identity("CREATE TABLE test_table (c1 INT, c2 INT) DUPLICATE KEY (c1)")

    def test_distributed(self):
        self.validate_identity(
            "CREATE TABLE test_table (c1 INT, c2 INT) UNIQUE KEY (c1) DISTRIBUTED BY HASH (c1)"
        )
        self.validate_identity("CREATE TABLE test_table (c1 INT, c2 INT) DISTRIBUTED BY RANDOM")
        self.validate_identity(
            "CREATE TABLE test_table (c1 INT, c2 INT) DISTRIBUTED BY RANDOM BUCKETS 1"
        )

    def test_partitionbyrange(self):
        self.validate_identity(
            "CREATE TABLE test_table (c1 INT, c2 DATE) PARTITION BY RANGE (`c2`) (PARTITION `p201701` VALUES LESS THAN ('2017-02-01'), PARTITION `p201702` VALUES LESS THAN ('2017-03-01'))"
        )
        self.validate_identity(
            "CREATE TABLE test_table (c1 INT, c2 DATE) PARTITION BY RANGE (`c2`) (PARTITION `p201701` VALUES [('2017-01-01'), ('2017-02-01')), PARTITION `other` VALUES LESS THAN (MAXVALUE))"
        )
        self.validate_identity(
            "CREATE TABLE test_table (c1 INT, c2 DATE) PARTITION BY RANGE (`c2`) (FROM ('2000-11-14') TO ('2021-11-14') INTERVAL 2 YEAR)"
        )

    def test_greenplum_to_doris_altertable(self):
        """Test conversion from Greenplum SQL to Doris SQL using direct SQL comparison."""

        # Test cases: input Greenplum SQL -> expected Doris SQL
        test_cases = [
            (
                "ALTER TABLE sales DROP PARTITION FOR (DATE'2023-01-01')",
                "ALTER TABLE sales DROP PARTITION p20230101",
            ),
            (
                "ALTER TABLE sales TRUNCATE PARTITION FOR (DATE'2023-01-01')",
                "TRUNCATE TABLE sales PARTITION(p20230101)",
            ),
            (
                "ALTER TABLE sales DROP PARTITION IF EXISTS FOR (DATE'2023-01-01')",
                "ALTER TABLE sales DROP PARTITION IF EXISTS p20230101",
            ),
            (
                "ALTER TABLE sales ADD PARTITION p1 START (DATE'2023-01-01') INCLUSIVE END (DATE'2023-02-01') EXCLUSIVE",
                "ALTER TABLE sales ADD PARTITION p1 VALUES [('2023-01-01'), ('2023-02-01'))",
            ),
            (
                "ALTER TABLE sales ADD PARTITION p2 START (DATE'2023-01-01') INCLUSIVE END (DATE'2023-02-01') EXCLUSIVE WITH (tablespace='ts1')",
                "ALTER TABLE sales ADD PARTITION p2 VALUES [('2023-01-01'), ('2023-02-01'))",
            ),
            (
                "ALTER TABLE sales ADD PARTITION IF NOT EXISTS p3 START (DATE'2023-01-01') INCLUSIVE END (DATE'2023-02-01') EXCLUSIVE",
                "ALTER TABLE sales ADD PARTITION IF NOT EXISTS p3 VALUES [('2023-01-01'), ('2023-02-01'))",
            ),
            (
                "ALTER TABLE sales ADD PARTITION p4 END (DATE'2023-02-01') EXCLUSIVE",
                "ALTER TABLE sales ADD PARTITION p4 VALUES LESS THAN ('2023-02-01')",
            ),
        ]

        for greenplum_sql, expected_doris_sql in test_cases:
            with self.subTest(greenplum_sql=greenplum_sql):
                # Parse Greenplum SQL and convert to Doris
                parsed = sqlglot.parse(greenplum_sql, dialect="greenplum")[0]
                actual_doris_sql = parsed.sql(
                    dialect="doris", unsupported_level=sqlglot.ErrorLevel.IGNORE
                )

                # Compare with expected Doris SQL
                self.assertEqual(actual_doris_sql, expected_doris_sql)

    def test_greenplum_to_doris_altertable_unsupported_cases(self):
        """Test that unsupported Greenplum partition syntax raises errors when converting to Doris."""
        unsupported_cases = [
            "ALTER TABLE sales ADD PARTITION p2 START (DATE'2023-02-01') INCLUSIVE END (DATE'2023-03-01') INCLUSIVE",
            "ALTER TABLE sales ADD PARTITION p3 START (DATE'2023-03-01') EXCLUSIVE END (DATE'2023-04-01') EXCLUSIVE",
            "ALTER TABLE sales ADD PARTITION p4 START (DATE'2023-04-01') EXCLUSIVE END (DATE'2023-05-01') INCLUSIVE",
            "ALTER TABLE sales ADD PARTITION p5 START (DATE'2023-05-01') INCLUSIVE",
            "ALTER TABLE sales ADD PARTITION p7 START (DATE'2023-07-01') EXCLUSIVE",
            "ALTER TABLE sales ADD PARTITION p8 END (DATE'2023-08-01') INCLUSIVE",
        ]

        for sql in unsupported_cases:
            with self.subTest(sql=sql):
                # Parse with Greenplum dialect
                parsed = sqlglot.parse(sql, dialect="greenplum")[0]

                # We expect conversion to Doris to raise an exception
                with self.assertRaises(Exception) as context:
                    parsed.sql(dialect="doris")

                # Verify it's the expected Doris conversion error
                self.assertIn("Doris partition conversion error", str(context.exception))

    def test_greenplum_to_doris_table_alias_conversion(self):
        """Test conversion from Greenplum to Doris for DELETE/UPDATE statements with table aliases."""

        # Test cases for DELETE statements with table aliases
        self.validate_all(
            "DELETE FROM sales AS s WHERE s.id = 1",
            write={
                "doris": "DELETE FROM sales s WHERE s.id = 1",
                "greenplum": "DELETE FROM sales AS s WHERE s.id = 1",
            },
        )

        # DELETE with multiple table references
        self.validate_all(
            "DELETE FROM orders AS o WHERE o.customer_id IN (SELECT c.id FROM customers AS c WHERE c.status = 'inactive')",
            write={
                "doris": "DELETE FROM orders o WHERE o.customer_id IN (SELECT c.id FROM customers c WHERE c.`status` = 'inactive')",
                "greenplum": "DELETE FROM orders AS o WHERE o.customer_id IN (SELECT c.id FROM customers AS c WHERE c.status = 'inactive')",
            },
        )

        # DELETE with EXISTS clause
        self.validate_all(
            "DELETE FROM temp_data AS t WHERE NOT EXISTS (SELECT 1 FROM main_data AS m WHERE m.id = t.id)",
            write={
                "doris": "DELETE FROM temp_data t WHERE NOT EXISTS(SELECT 1 FROM main_data m WHERE m.id = t.id)",
                "greenplum": "DELETE FROM temp_data AS t WHERE NOT EXISTS(SELECT 1 FROM main_data AS m WHERE m.id = t.id)",
            },
        )

        # UPDATE statements with table aliases
        self.validate_all(
            "UPDATE employees AS e SET e.salary = e.salary * 1.1 WHERE e.department = 'IT'",
            write={
                "doris": "UPDATE employees e SET e.salary = e.salary * 1.1 WHERE e.department = 'IT'",
                "greenplum": "UPDATE employees AS e SET e.salary = e.salary * 1.1 WHERE e.department = 'IT'",
            },
        )

        # UPDATE with subquery
        self.validate_all(
            "UPDATE inventory AS i SET i.quantity = 0 WHERE i.product_id IN (SELECT p.id FROM products AS p WHERE p.discontinued = true)",
            write={
                "doris": "UPDATE inventory i SET i.quantity = 0 WHERE i.product_id IN (SELECT p.id FROM products p WHERE p.discontinued = TRUE)",
                "greenplum": "UPDATE inventory AS i SET i.quantity = 0 WHERE i.product_id IN (SELECT p.id FROM products AS p WHERE p.discontinued = TRUE)",
            },
        )

        # UPDATE with multiple columns
        self.validate_all(
            "UPDATE accounts AS a SET a.balance = a.balance + 100, a.status = 'active' WHERE a.account_type = 'savings'",
            write={
                "doris": "UPDATE accounts a SET a.balance = a.balance + 100, a.`status` = 'active' WHERE a.account_type = 'savings'",
                "greenplum": "UPDATE accounts AS a SET a.balance = a.balance + 100, a.status = 'active' WHERE a.account_type = 'savings'",
            },
        )

        # UPDATE with CASE statement
        self.validate_all(
            "UPDATE users AS u SET u.status = CASE WHEN u.age >= 18 THEN 'adult' ELSE 'minor' END",
            write={
                "doris": "UPDATE users u SET u.`status` = CASE WHEN u.age >= 18 THEN 'adult' ELSE 'minor' END",
                "greenplum": "UPDATE users AS u SET u.status = CASE WHEN u.age >= 18 THEN 'adult' ELSE 'minor' END",
            },
        )

        # UPDATE with multiple table references in subquery
        self.validate_all(
            "UPDATE prices AS p SET p.amount = p.amount * 0.9 WHERE p.product_id IN (SELECT pr.id FROM products AS pr JOIN categories AS c ON pr.category_id = c.id WHERE c.name = 'Electronics')",
            write={
                "doris": "UPDATE prices p SET p.amount = p.amount * 0.9 WHERE p.product_id IN (SELECT pr.id FROM products pr JOIN categories c ON pr.category_id = c.id WHERE c.`name` = 'Electronics')",
                "greenplum": "UPDATE prices AS p SET p.amount = p.amount * 0.9 WHERE p.product_id IN (SELECT pr.id FROM products AS pr JOIN categories AS c ON pr.category_id = c.id WHERE c.name = 'Electronics')",
            },
        )

        # UPDATE with simple assignment
        self.validate_all(
            "UPDATE customers AS c SET c.full_name = 'John Doe' WHERE c.full_name IS NULL",
            write={
                "doris": "UPDATE customers c SET c.full_name = 'John Doe' WHERE c.full_name IS NULL",
                "greenplum": "UPDATE customers AS c SET c.full_name = 'John Doe' WHERE c.full_name IS NULL",
            },
        )
