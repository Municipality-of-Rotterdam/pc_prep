"""Pytest can make use of fixtures to for example use test data.

For testing it is important that your test data is predictable.

This is where you can define fixtures that should be available across test files.
It is also possible to create fixtures within the test_file (e.g. example_test)
but these fixtures are only locally available.

In the fixtures below we use a sqllite database, but you could use any database or datastructure
as long as it is available at time of testing.
"""

import pytest
from sqlalchemy import Connection, Engine, create_engine, text


@pytest.fixture
def sql_connection_engine():
    engine = create_engine("sqlite://", echo=True)
    return engine

def create_sql(openbare_ruimte_id: str, 
               nummeraanduiding_id: str,
               woonplaats_id: str,
               naam_van_openbare_ruimte: str,
               huisnummer: int,
               postcode: str,
               huisletter: str,
               huisnummertoevoeging: str,
               begin_geldigheid: str,
               eind_geldigheid: str,
               current_indicator: int):
    return f"""INSERT INTO [Current].[LVBAG2_0_ADRES] (
        openbare_ruimte_id, 
        nummeraanduiding_id,
        woonplaats_id,
        naam_van_openbare_ruimte,
        huisnummer,
        postcode,
        huisletter,
        huisnummertoevoeging,
        begin_geldigheid,
        eind_geldigheid,
        current_indicator) VALUES (
        '{openbare_ruimte_id}', 
        '{nummeraanduiding_id}',
        '{woonplaats_id}',
        '{naam_van_openbare_ruimte}',
        {huisnummer},
        '{postcode}',
        '{huisletter}',
        '{huisnummertoevoeging}',
        '{begin_geldigheid}',
        '{eind_geldigheid}',
        {current_indicator})"""

@pytest.fixture
def sql_connection(sql_connection_engine: Engine):
    connection = sql_connection_engine.connect()
    
    yield connection

    connection.close()

@pytest.fixture
def setupdb(sql_connection: Connection):
    sql_connection.execute(text("ATTACH DATABASE ':memory:' as Current"))
    sql_connection.execute(text("""CREATE TABLE [Current].[LVBAG2_0_ADRES] (
                                openbare_ruimte_id TEXT NOT NULL, 
                                nummeraanduiding_id TEXT,
                                woonplaats_id TEXT NOT NULL,
                                naam_van_openbare_ruimte TEXT NOT NULL,
                                huisnummer NUMERIC NOT NULL,
                                postcode TEXT NOT NULL,
                                huisletter TEXT NOT NULL,
                                huisnummertoevoeging TEXT NOT NULL,
                                begin_geldigheid TEXT NOT NULL,
                                eind_geldigheid TEXT NOT NULL,
                                current_indicator NUMERIC NOT NULL)"""))
    sql_connection.execute(text(create_sql('0599102545875', 'a', '05991', 'PARK', 31, '3071WK', None, None, '01-01-2000', '02-02-2040', 1)))
    sql_connection.execute(text(create_sql('0599102545876', ' ', '05991', 'PARK', 29, '3071WK', 'a', None, '01-01-2000', '02-02-2040', 1)))
    sql_connection.execute(text(create_sql('0599102545877', 'bb', '05991', 'PARK', 2828, '3071WK', 'b', None, '01-01-2000', '02-02-2040', 1)))
    sql_connection.execute(text(create_sql('0599102545878', '', '05991', 'PARK', 130, '3071WK', None, 'b', '01-01-2000', '02-02-2040', 1)))
    sql_connection.execute(text(create_sql('0599102545879', 'a', '05991', 'PARK', 50, '3071WK', None, 'c', '01-01-2000', '02-02-2040', 1)))
    sql_connection.execute(text(create_sql('0599102545870', '', '05991', 'PARK', 0, '3071WK', None, None, '01-01-2000', '02-02-2040', 1)))
    sql_connection.commit()
    
@pytest.fixture
def empty_db(sql_connection: Connection):
    sql_connection.execute(text("ATTACH DATABASE ':memory:' as Current"))
    sql_connection.execute(text("""CREATE TABLE [Current].[LVBAG2_0_ADRES] (
                                openbare_ruimte_id TEXT NOT NULL, 
                                nummeraanduiding_id TEXT,
                                woonplaats_id TEXT NOT NULL,
                                naam_van_openbare_ruimte TEXT NOT NULL,
                                huisnummer NUMERIC NOT NULL,
                                postcode TEXT NOT NULL,
                                huisletter TEXT NOT NULL,
                                huisnummertoevoeging TEXT NOT NULL,
                                begin_geldigheid TEXT NOT NULL,
                                eind_geldigheid TEXT NOT NULL,
                                current_indicator NUMERIC NOT NULL)"""))
    sql_connection.commit()
