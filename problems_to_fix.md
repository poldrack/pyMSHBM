# Development scratchpad

- Use this file to keep notes on ongoing development work.
- Open problems marked with [ ]
- Fixed problems marked with [x]

## NOTES

[x] Step 9 crashed with the following error, seems like it doesn't realize that the medial wall vertices have been removed:

```
INFO: Step 9/10: Running MRF-regularized individual parcellation (c=50.0, w=200.0)
INFO:   sub-s03 (12 sessions)
Traceback (most recent call last):
  File "/Users/poldrack/Dropbox/code/MS-HBM/pyMSHBM/.venv/bin/pymshbm-wrapper", line 10, in <module>
    sys.exit(main())
             ~~~~^^
  File "/Users/poldrack/Dropbox/code/MS-HBM/pyMSHBM/src/pymshbm/cli/wrapper.py", line 78, in main
    result_dir = run_wrapper(
        sub_list=args.sub_list,
    ...<12 lines>...
        overwrite_kmeans=args.overwrite_kmeans,
    )
  File "/Users/poldrack/Dropbox/code/MS-HBM/pyMSHBM/src/pymshbm/pipeline/wrapper.py", line 590, in run_wrapper
    lh_labels, rh_labels = parcellation_single_subject(
                           ~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        data=sub_data,
        ^^^^^^^^^^^^^^
    ...<4 lines>...
        max_iter=max_iter,
        ^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/poldrack/Dropbox/code/MS-HBM/pyMSHBM/src/pymshbm/pipeline/single_subject.py", line 39, in parcellation_single_subject
    labels = generate_individual_parcellation(
        group_priors=group_priors,
    ...<4 lines>...
        max_iter=max_iter,
    )
  File "/Users/poldrack/Dropbox/code/MS-HBM/pyMSHBM/src/pymshbm/core/individual.py", line 69, in generate_individual_parcellation
    s_lambda, kappa, s_t_nu = _vmf_em(
                              ~~~~~~~^
        data, s_lambda, kappa, s_t_nu, s_psi, sigma, theta,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        neighborhood, v_same, v_diff, w, c, dim, L, ini_concentration,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        valid_mask,
        ^^^^^^^^^^^
    )
    ^
  File "/Users/poldrack/Dropbox/code/MS-HBM/pyMSHBM/src/pymshbm/core/individual.py", line 205, in _vmf_em
    V_lam = v_lambda_product(neighborhood, v_same, v_diff, lam_active)
  File "/Users/poldrack/Dropbox/code/MS-HBM/pyMSHBM/src/pymshbm/math/mrf.py", line 42, in v_lambda_product
    lam_nb = lam[nb_valid, :]  # (n_valid, K)
             ~~~^^^^^^^^^^^^^
IndexError: index 80951 is out of bounds for axis 0 with size 74947
```

[ ] Step 8 should save its results so that they can be reused (unless --overwrite-em is specified)