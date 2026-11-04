import pytest
from lxml import etree
import preprocessing.sec_form4_collector as sec_collector

def test_sec_form4_module_imports_without_exit():
    """Verify sec_form4_collector imports cleanly without triggering SystemExit."""
    assert hasattr(sec_collector, "validate_user_agent")
    assert hasattr(sec_collector, "main")
    assert hasattr(sec_collector, "local_text")

def test_sec_form4_main_validates_credentials(monkeypatch):
    """Verify that execution without valid User-Agent raises SystemExit."""
    monkeypatch.setenv("SEC_USER_AGENT", "")
    with pytest.raises(SystemExit, match="Set SEC_USER_AGENT"):
        sec_collector.main("AAPL", limit=5)

def test_sec_form4_xml_parsing():
    """Verify XML parsing on synthetic Form 4 document structure."""
    xml_content = """<?xml version="1.0"?>
    <ownershipDocument>
        <issuer>
            <issuerTradingSymbol>MOCK</issuerTradingSymbol>
            <issuerName>Mock Corporation</issuerName>
        </issuer>
        <reportingOwner>
            <reportingOwnerId>
                <rptOwnerName>Jane Doe</rptOwnerName>
            </reportingOwnerId>
            <reportingOwnerRelationship>
                <isOfficer>1</isOfficer>
                <officerTitle>Chief Financial Officer</officerTitle>
            </reportingOwnerRelationship>
        </reportingOwner>
        <nonDerivativeTable>
            <nonDerivativeTransaction>
                <transactionCoding>
                    <transactionCode>S</transactionCode>
                </transactionCoding>
                <transactionAmounts>
                    <transactionShares><value>5000</value></transactionShares>
                    <transactionPricePerShare><value>150.00</value></transactionPricePerShare>
                    <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
                </transactionAmounts>
                <postTransactionAmounts>
                    <sharesOwnedFollowingTransaction><value>25000</value></sharesOwnedFollowingTransaction>
                </postTransactionAmounts>
                <transactionDate><value>2024-01-15</value></transactionDate>
            </nonDerivativeTransaction>
        </nonDerivativeTable>
    </ownershipDocument>
    """
    root = etree.fromstring(xml_content.encode("utf-8"))
    symbol = sec_collector.local_text(root, "//*[local-name()='issuerTradingSymbol']")
    issuer = sec_collector.local_text(root, "//*[local-name()='issuerName']")
    owner = sec_collector.local_text(root, "//*[local-name()='rptOwnerName']")
    code = sec_collector.local_text(root, "//*[local-name()='transactionCode']")
    shares = float(sec_collector.local_text(root, "//*[local-name()='transactionShares']/*[local-name()='value']"))
    price = float(sec_collector.local_text(root, "//*[local-name()='transactionPricePerShare']/*[local-name()='value']"))
    after = float(sec_collector.local_text(root, "//*[local-name()='sharesOwnedFollowingTransaction']/*[local-name()='value']"))

    assert symbol == "MOCK"
    assert issuer == "Mock Corporation"
    assert owner == "Jane Doe"
    assert code == "S"
    assert shares == 5000.0
    assert price == 150.0
    assert after == 25000.0

