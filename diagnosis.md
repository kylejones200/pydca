# CI Diagnosis and Fix Plan

## Errors Identified

### 1. **Import Error: Missing `Optional` in `multiphase.py`** ⚠️ CRITICAL
   - **File**: [`decline_analysis/multiphase.py`](decline_analysis/multiphase.py:11)
   - **Lines**: 31, 32, 33, 34, 124, 292, 293, 295
   - **Error**: `NameError: name 'Optional' is not defined` / `F821 undefined name 'Optional'`
   - **Root Cause**: `Optional` was removed from imports when cleaning up unused imports, but it's actually used in type hints throughout the file
   - **Fix**: Add `from typing import Optional` to line 11 in `multiphase.py`

## Summary

**Total Errors**: 1 critical error (blocks all tests)

**Priority**: Critical - Tests cannot run without this fix

**Estimated Fix Time**: < 1 minute

**Impact**: 
- All tests fail to collect due to import error
- Flake8 reports 8 undefined name errors
- CI will fail on test job

## Fix Plan

1. **Fix**: Add `Optional` import to `decline_analysis/multiphase.py`
2. **Verify**: Run `pytest tests/ -v --maxfail=1` locally
3. **Verify**: Run `flake8 decline_analysis/ tests/ --select=F821` locally
4. **Create**: Branch `fix/multiphase-optional-import`
5. **Commit**: `multiphase: add missing Optional import`
6. **Push**: Branch and create PR
