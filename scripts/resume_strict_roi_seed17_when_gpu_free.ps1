throw (
    "This legacy GPU queue/resume helper is retired. It references checkpoints " +
    "from the failed strict_roi_results namespace, which cannot enter the clean " +
    "v1.1.1 study. Resume only a v1.1.1 clean run with " +
    "scripts\\run_strict_roi_clean_seed.ps1 -Seed 17 -ResumeIncompleteSeed."
)
