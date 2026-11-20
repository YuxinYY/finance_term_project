# DRAFT — not sent. Humans review and send.

**To:** Roger Stein; [TA]
**Subject:** Team Turquoise — five quick questions on the PD term project data

Dear Professor Stein and [TA],

We are Team Turquoise (Kiet Nguyen, Desmond Peng, Chloe Li). We are finalising our formulation
and have five short questions. For each we state the assumption we will use if we do not hear back,
so a one-word "fine" is enough.

1. **Default label window.** For a statement with `stmt_date` *s*, which 12-month window defines a
   default for the holdout evaluation — e.g. defaults in (*s*, *s*+12m], or a window that starts
   after the statement could realistically have been filed (e.g. *s*+6m to *s*+18m)?
   *Default assumption:* we will choose a window based on our data analysis, document it, and
   show how sensitive our results are to the window start.

2. **Defaults dated 2013.** Appendix B says the training data has no information on defaults after
   12/31/2012, but `def_date` in `train-2025.csv` contains dates in 2013. Should we treat these
   2013 dates as valid default events, or ignore them?
   *Default assumption:* we will report how many rows this affects and use the dates as given,
   with a sensitivity check that drops them.

3. **Holdout period.** Is the holdout a single future fiscal year (e.g. `fs_year` 2013) or several?
   *Default assumption:* a single following year.

4. **Firm history in the harness.** When a holdout firm also appears in the training data, may the
   prediction code use a frozen lookup of that firm's *training-data* statements (e.g. the prior
   year's ratios), as long as it never uses other holdout rows?
   *Default assumption:* no. We will use only the current record, so every firm is treated the
   same way whether or not we have seen it before.

5. **External macro data.** Since there is little time before Nov 16, we assume no external data
   (e.g. Italian unemployment) can be used, and we will handle the economic cycle through how we
   set the overall PD level. Please let us know if that is wrong.

Thank you,
Team Turquoise
