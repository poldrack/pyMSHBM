# Development scratchpad

- Use this file to keep notes on ongoing development work.
- Open problems marked with [ ]
- Fixed problems marked with [x]

## NOTES

[x] There seems to be a problem with the way that the medial wall vertices are being determined.  I ran it and got the following output:

```
INFO: Step 6/10: Loading profiles into training tensor
INFO:   81924/81924 vertices are non-medial-wall (0% reduction)
```

**Fix**: Added `load_cortex_mask()` to load FreeSurfer `{hemi}.cortex.label` files and zero out medial wall vertex profiles before normalization in `_load_profiles_tensor()`, matching MATLAB CBIG behavior. Falls back to zero-profile heuristic when cortex labels are unavailable.
