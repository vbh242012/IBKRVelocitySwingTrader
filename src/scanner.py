from ib_async import ScannerSubscription

from src.config import (
    IB_SCANNER_SCAN_CODE,
    IB_SCANNER_SCAN_CODES,
    IB_SCANNER_LOCATION_CODE,
    IB_SCANNER_ROWS,
    SCAN_MIN_PRICE,
    SCAN_MIN_VOLUME,
    SCAN_MIN_MKTCAP,
)


def build_momentum_scanner() -> ScannerSubscription:
    """IB scanner: broad active corporate stocks; custom rules decide momentum."""
    return ScannerSubscription(
        numberOfRows=IB_SCANNER_ROWS,
        instrument='STK',
        locationCode=IB_SCANNER_LOCATION_CODE,
        scanCode=IB_SCANNER_SCAN_CODE,
        abovePrice=SCAN_MIN_PRICE,
        aboveVolume=int(SCAN_MIN_VOLUME),
        marketCapAbove=SCAN_MIN_MKTCAP / 1_000_000,  # IB field is in millions
        stockTypeFilter='CORP',                        # exclude ETFs at scanner level
    )


def build_momentum_scanners() -> list:
    """One ScannerSubscription per configured scan code; caller deduplicates results."""
    return [
        ScannerSubscription(
            numberOfRows=IB_SCANNER_ROWS,
            instrument='STK',
            locationCode=IB_SCANNER_LOCATION_CODE,
            scanCode=code,
            abovePrice=SCAN_MIN_PRICE,
            aboveVolume=int(SCAN_MIN_VOLUME),
            marketCapAbove=SCAN_MIN_MKTCAP / 1_000_000,
            stockTypeFilter='CORP',
        )
        for code in IB_SCANNER_SCAN_CODES
    ]
