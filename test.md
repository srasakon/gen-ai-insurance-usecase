# Field: `total_assets`

- **Node ID:** FIN_11
- **Scope:** [Balance sheet, Audited or reviewed financial statements, Compiled or internally prepared financial statements, Tax-basis or cash-basis financial statements, Business income tax return, Exempt organization return]
- **Product line:** [Casualty, Inland Marine, Life Sciences, Property, Railroad, Transportation, Wholesale Fin Lines, Small Fin Lines]

## Definition
The total of everything the reporting entity owns at one balance sheet date, copied from the printed total-assets line. It sizes the entity for management liability rating and appetite bands, and anchors leverage and financial-strength review. It is a point-in-time stock, never a period flow.

## Is
- The line labelled "Total assets", "TOTAL ASSETS", "Total Assets" or "Assets, total" on a statement titled balance sheet, consolidated balance sheet, statement of financial position, statement of financial condition, statement of condition, statement of net position (government), statement of assets, liabilities and equity (income tax basis), statement of assets, liabilities and net assets (modified cash basis), or statement of assets and liabilities arising from cash transactions.
- Both layouts: assets stacked above liabilities, or assets on the left half of the page and liabilities and equity on the right; the total sits at the foot of the assets side.
- Not-for-profit statements of financial position: "Total assets" printed above liabilities and net assets.
- Government statements of net position: the "Total assets" line printed before the deferred outflows of resources section.
- Side-by-side entity columns: the column headed "Consolidated", "Consolidated total", "Combined", "Total" or "Total (memorandum only)".
- Column headings that set the date: "December 31, 20XX", "As of", "At", "June 30, 20XX and 20XX", "Fiscal year end", "53 weeks ended", "Successor" / "Predecessor".
- A business tax return supplied without statements: the balance-sheet schedule (Schedule L) total assets, end-of-tax-year column (d) — line 15 on corporate and S corporation returns, line 14 on partnership returns — or, failing that, the total-assets box in the return header.
- An exempt organization return supplied without statements: the balance-sheet total assets, end-of-year column (Part X line 16 on the full return; line 25 on the short return).

## Is not
- "Total liabilities and equity", "Total liabilities and stockholders' (shareholders', members', partners') equity", "Total liabilities and capital", "Total liabilities and net assets", "Total liabilities, deferred inflows and net position": equal in amount, but a different line.
- "Net assets", "Total net assets", "Net assets without/with donor restrictions", "Net position", "Net worth": these are equity, not assets.
- Balance sheet formats that never print a total: "Total assets less current liabilities", "Net current assets (liabilities)", "Fixed assets" plus "Current assets" (common in UK and Irish statutory accounts).
- "Total assets and deferred outflows of resources" on a government statement.
- Subtotals: total current assets, total noncurrent (long-term, fixed, other) assets, property and equipment net, total investments, assets held for sale, assets of discontinued operations, restricted cash.
- Assets of one subsidiary, segment, fund, division or the parent only; eliminations or adjustments columns; segment-note totals.
- The beginning-of-year column (b) of a tax return balance sheet; prior-year comparatives returned as the current figure (return them as their own dated objects).
- Off-balance-sheet amounts: assets under management or administration, custodial or fiduciary funds held for clients, benefit plan assets ("net assets available for benefits"), escrow balances disclosed only in the notes.
- Insured values, statement-of-values totals, appraisals, fair market or liquidation values, enterprise value, market capitalization, pro forma, projected or budgeted balance sheets.

## Rules
- Return an array of objects, newest date first: as_of (YYYY-MM-DD from the column heading), amount (number), currency (ISO code if stated or shown by symbol; else null), label (as printed), entity_scope (consolidated, combined, single_entity, unspecified), source_document (as printed title), basis_of_accounting (gaap, ifrs, income_tax, cash, modified_cash, statutory, unspecified), assurance (audited, reviewed, compiled, internal, tax_return, unspecified), restated (true/false), units_note (as printed or null).
- Copy the figure only from a line whose label reads total assets. Never add subtotals, never compute liabilities plus equity, never compute fixed plus current assets: a computed number hides a misread or cropped page.
- Units: apply the scale printed in the statement heading, column heading or a note on the same page ("in thousands", "(000s)", "$000", "000's omitted", "in millions", "$M", "(Dollars in millions)", "€m"), exactly once, and copy the wording to units_note. A scale printed only for a different table does not apply.
- Strip currency symbols, commas, spaces and footnote markers ("1", "(a)", "*"). A total-assets line shown as a dash or blank means found: false for that date.
- Entity columns: take the consolidated or combined total. With several entity columns and no total, return one object per entity with an entity key; never add them.
- Successor and predecessor columns separated by a vertical line are different reporting entities: return each date as its own object; never merge.
- A column marked "as restated" or "revised" replaces the original for that date and sets restated true; a "pro forma" column is never returned.
- Several documents for the same date: keep one, preferring audited, then reviewed, then compiled, then internal, then tax return; between equals, the later-issued one.
- If no total-assets line is printed for any date, return found: false.

## Evidence to weigh
For each candidate: statement title | label as printed | column heading and date | entity scope of the column | units note and where it is printed | assets side or liabilities side | pro forma, projected or actual | document assurance | keep/strike with reason.

## Verify before answering
- The quoted label says total assets and sits on the assets side, not the liabilities-and-equity or net-assets side.
- as_of comes from that column's own heading; comparative columns are separate objects.
- The scale was applied once and matches units_note.
- The column covers the whole reporting entity.

## Examples
### Example 1
- **Situation:** A comparative balance sheet headed "(in thousands)" shows total assets and, at the foot, total liabilities and stockholders' equity with identical figures.
- **Method:** Take the assets-side line; apply thousands once; each date its own object.
- **Result:** Two objects, newest first, units_note "in thousands"

### Example 2
- **Situation:** A not-for-profit statement of financial position prints total assets, then total liabilities, then total net assets, which is a smaller number.
- **Method:** Net assets is equity; total assets is the assets line.
- **Result:** One object with the total-assets amount

### Example 3
- **Situation:** Statutory accounts show fixed assets, current assets, net current assets and total assets less current liabilities, but no total-assets line.
- **Method:** No line reads total assets; adding fixed and current assets is computing.
- **Result:** found: false

---

# Field: `annual_revenue`

- **Node ID:** FIN_12
- **Scope:** [Income statement, Audited or reviewed financial statements, Compiled or internally prepared financial statements, Tax-basis or cash-basis financial statements, Business income tax return, Exempt organization return]
- **Product line:** [Casualty, Inland Marine, Life Sciences, Middle Market Casualty, Middle Market Property, Property, Railroad, Small Fin Lines, Transportation, Wholesale Fin Lines]

## Definition
One figure: the entity's top-line revenue for its most recent completed fiscal year, as printed. It is the exposure and size measure underwriters rate on (gross sales for products and premises classes, revenue bands for management, professional and cyber lines, revenue thresholds for appetite), so it must be twelve months, actual, historical and whole-entity.

## Is
- The top revenue line of a statement titled income statement, statement of income, statement of operations, statement of earnings, profit and loss (P&L), profit and loss account, statement of comprehensive income, statement of profit or loss, statement of revenues and expenses (income tax, cash or modified cash basis), statement of revenue collected and expenses paid, statement of activities (not-for-profit), or statement of revenues, expenses and changes in net position (government business-type).
- General labels: "Total revenue(s)", "Revenue(s)", "Net revenue(s)", "Total net revenue", "Net sales", "Sales, net", "Sales", "Net sales and revenues", "Operating revenue(s)", "Total operating revenue(s)", "Revenue from contracts with customers", "Turnover" (UK, Ireland), "Revenue from operations".
- Industry labels, when they are the top revenue total:
  - Contractors: "Contract revenue(s)", "Contract revenues earned", "Revenues earned on contracts", "Construction revenue".
  - Services and professional firms: "Fee income", "Fees earned", "Professional fees", "Service revenue", "Revenue from services", "Commission income", "Net commissions and fees".
  - Trucking, rail, logistics: "Operating revenues", "Total operating revenues", "Freight revenue(s)" totalled with other revenue, "Revenue, including fuel surcharge", "Total revenue" including brokerage.
  - Healthcare: "Net patient service revenue", "Patient service revenue, net", "Total operating revenue(s)", "Net operating revenue(s)".
  - Real estate and lessors: "Rental revenue", "Rental income", "Total rental revenue", "Total property revenues".
  - Hospitality: "Total revenues" over rooms, food and beverage and other departments.
  - Not-for-profits: "Total revenue, gains and other support", "Total support and revenue", "Total public support and revenue", "Total revenues, gains and other support", in the total column.
- Period headings meaning a full year: "Year ended", "Years ended", "Fiscal year ended", "Twelve months ended", "52 weeks ended", "53 weeks ended", "FY20XX".
- A business tax return supplied without statements: gross receipts or sales less returns and allowances — line 1c on corporate, S corporation and partnership returns; line 3 on a sole proprietor's business schedule. An exempt organization return: current-year total revenue (Part I line 12 on the full return; line 9 on the short return).
- A column labelled "trailing twelve months", "TTM", "LTM" or "last twelve months", only when no completed fiscal year is printed anywhere.

## Is not
- Interim or partial periods: "Three/six/nine months ended", "Quarter", "Month", "Year-to-date", "YTD", "Period from inception through", "Transition period" — and any figure annualized, run-rate, extrapolated or "estimated full year".
- Budget, forecast, projected, plan, pro forma, "as adjusted", variance, "increase (decrease)" and percentage-of-revenue columns.
- Lines above net: "Gross sales", "Gross revenue" before returns, allowances, discounts, rebates or chargebacks when a net line is printed; healthcare "Gross patient revenue", "Total patient revenue" or "Gross charges" before contractual allowances and implicit price concessions; "Gross billings" when the entity reports net revenue.
- Lines below or beside revenue: gross profit (gross margin), operating income, net income, EBITDA, "Other income", "Other operating income", "Non-operating income", interest income, dividend and investment income, gains on sale of assets, equity in earnings of affiliates, insurance recoveries, government grants or loan forgiveness shown as other income, and "Total income" that adds other income to revenue.
- Non-GAAP or partial revenue: "Revenue excluding fuel surcharge", "Freight revenue, excludes fuel surcharge", "Adjusted revenue", "Organic revenue", "Same-store sales", "Revenue per mile".
- Sales, excise, use or value-added taxes collected for a government shown on their own line; billings, bookings, backlog, orders, gross merchandise value, deferred or unearned revenue, unbilled receivables.
- Revenue of one segment, subsidiary, location, product line or discontinued operation; parent-only or eliminations columns.
- Cash-basis "receipts" that include loan proceeds, owner contributions or asset sales: only the receipts line for sales or services counts.
- A tax return's "Total income" or "Ordinary business income" line.

## Rules
- Return an object: amount (number), currency (ISO code if stated or shown by symbol; else null), period_end (YYYY-MM-DD from the heading), months (12, or the stated length when the fallback applies), label (as printed), entity_scope (consolidated, combined, single_entity, unspecified), source_document (as printed title), basis_of_accounting (gaap, ifrs, income_tax, cash, modified_cash, unspecified), assurance (audited, reviewed, compiled, internal, tax_return, unspecified), restated (true/false), units_note (as printed or null).
- Choose the most recent period of twelve months (52 or 53 weeks count as twelve) across all documents supplied. A newer interim column never replaces it.
- Fallback order when no twelve-month actual exists: a labelled trailing-twelve-month column; then the latest short fiscal year with its stated months. Never annualize, never add quarters or months.
- Take the top revenue total: a "Total" line wins over the component lines above it; net wins over gross; a line that adds other income is not the top revenue total.
- Not-for-profits: the total column (without plus with donor restrictions as printed together); never add "net assets released from restrictions", which nets to zero across columns.
- Successor and predecessor periods inside one fiscal year are not a twelve-month period and are never combined; use the latest full year that is not split.
- Units: apply the printed scale once ("in thousands", "(000s)", "$000", "000's omitted", "in millions", "$M", "€m") and copy it to units_note. Strip symbols, commas and footnote markers.
- Same fiscal year in several documents: prefer audited, then reviewed, then compiled, then internal, then tax return; a restated figure replaces the original.
- If no top revenue line for any qualifying period is printed, return found: false.

## Evidence to weigh
For each candidate: statement title | label as printed | position (top total, component, gross, below revenue) | column heading and period length | actual, budget, pro forma or interim | entity scope | units note | assurance and issue order | keep/strike with reason.

## Verify before answering
- The column heading states twelve months (or 52/53 weeks) of actual results, or months shows the fallback.
- The label is the top revenue total, net of returns, allowances and contractual adjustments, and excludes other income.
- No newer twelve-month period exists in any supplied document.
- The scale was applied once and matches units_note.

## Examples
### Example 1
- **Situation:** Audited statements for two fiscal years and an internal nine-month statement for the current year.
- **Method:** The nine-month column is interim; the latest full year wins.
- **Result:** The latest fiscal year's amount, months 12, assurance audited

### Example 2
- **Situation:** A trucking statement shows freight revenue, fuel surcharge revenue and total operating revenues; a summary table also shows revenue excluding fuel surcharge.
- **Method:** Total operating revenues is the top total; the excluding-surcharge figure is partial.
- **Result:** The total operating revenues amount

### Example 3
- **Situation:** A hospital statement shows gross patient revenue, deductions for contractual allowances, net patient service revenue, other revenue and total operating revenue.
- **Method:** Gross charges are above net; the top total includes other operating revenue.
- **Result:** The total operating revenue amount

---

# Field: `total_revenue`

- **Node ID:** FIN_04
- **Scope:** [Income statement, Audited or reviewed financial statements, Compiled or internally prepared financial statements, Tax-basis or cash-basis financial statements, Interim or management financial reports]
- **Product line:** [Casualty, Inland Marine, Life Sciences, Middle Market Casualty, Middle Market Property, Property, Railroad, Small Fin Lines, Transportation, Wholesale Fin Lines]

## Definition
The top-line revenue total the income statement reports for every actual period it presents — annual, comparative, interim and year-to-date — each tagged with its period. It shows growth, decline, seasonality and trend for financial-strength and exposure review.

## Is
- The top revenue total of a statement titled income statement, statement of income, statement of operations, statement of earnings, profit and loss (P&L), profit and loss account, statement of comprehensive income, statement of profit or loss, statement of revenues and expenses (income tax, cash or modified cash basis), statement of activities (not-for-profit), statement of revenues, expenses and changes in net position (government business-type), or a management or monthly financial report.
- General labels: "Total revenue(s)", "Revenue(s)", "Net revenue(s)", "Total net revenue", "Net sales", "Sales, net", "Sales", "Net sales and revenues", "Operating revenue(s)", "Total operating revenue(s)", "Revenue from contracts with customers", "Turnover", "Revenue from operations".
- Industry labels, when they are the top revenue total: "Contract revenue(s)", "Contract revenues earned" (contractors); "Fee income", "Fees earned", "Service revenue", "Commission income" (services); "Operating revenues", "Total operating revenues", "Freight revenue(s)" totalled with other revenue (trucking, rail, logistics); "Net patient service revenue", "Total operating revenue(s)" (healthcare); "Rental revenue", "Total rental revenue" (lessors); "Total revenue, gains and other support", "Total support and revenue" (not-for-profits, total column).
- Period headings: "Year(s) ended", "Fiscal year ended", "Twelve months ended", "52/53 weeks ended", "Three/six/nine months ended", "Quarter ended", "Month of", "Year-to-date", "YTD", "Period from [date] (inception) through [date]", "Trailing twelve months", "Successor" / "Predecessor".

## Is not
- Budget, forecast, plan, projected, pro forma, "as adjusted", prior-year-budget, variance, "increase (decrease)", "% change" and percentage-of-revenue columns.
- Lines above net: "Gross sales", "Gross revenue" before returns, allowances, discounts and rebates when a net line is printed; "Gross patient revenue", "Total patient revenue" or "Gross charges" before contractual allowances; "Gross billings" when net revenue is reported.
- Lines below or beside revenue: gross profit, operating income, net income, EBITDA, "Other income", "Other operating income", "Non-operating income", interest, dividend and investment income, gains on asset sales, equity in earnings of affiliates, grants or loan forgiveness shown as other income, and "Total income" that adds other income to revenue.
- Partial or non-GAAP measures: "Revenue excluding fuel surcharge", "Adjusted revenue", "Organic revenue", "Same-store sales", run-rate or annualized figures.
- Sales, excise or value-added taxes collected for a government shown on their own line; billings, bookings, backlog, orders, deferred revenue.
- Revenue of one segment, subsidiary, location or discontinued operation; parent-only or eliminations columns.
- Cash-basis receipts from loans, owner contributions or asset sales.

## Rules
- Return an array of objects, one per actual-results column, newest period end first and longer period first on the same date: period_start (YYYY-MM-DD if printed, else null), period_end (YYYY-MM-DD), months (from the heading: "year" 12, "52/53 weeks" 12, "three months" 3, "nine months" 9, "month of" 1; null if not stated), period_type (annual, interim, ytd, ttm, short_period), label (as printed), amount (number), currency (ISO code or null), entity_scope (consolidated, combined, single_entity, unspecified), restated (true/false), units_note (as printed or null).
- Copy every actual column, comparative years included. Never add months, quarters or successor and predecessor periods together; never annualize; never derive a column by subtraction.
- Take the top revenue total: a "Total" line wins over its components; net wins over gross; a line that adds other income is not the top revenue total.
- Not-for-profits: the total column only; never add "net assets released from restrictions".
- A column marked "as restated" or "revised" replaces the original for that period and sets restated true.
- The same period in several documents: keep one, preferring audited, then reviewed, then compiled, then internal; between equals, the later-issued one.
- Units: apply the printed scale once ("in thousands", "(000s)", "$000", "000's omitted", "in millions", "$M", "€m"); strip symbols, commas and footnote markers.
- If no top revenue line is printed for any actual period, return found: false.

## Evidence to weigh
For each column: statement title | heading as printed | period length and type | actual, budget, pro forma or variance | label and its position (top total, component, gross, below revenue) | entity scope | units note | restated or original | keep/strike with reason.

## Verify before answering
- Every object comes from an actual-results column with its own heading; no budget or variance column is returned.
- months and period_type match the heading wording, never an assumption.
- Each amount is the top revenue total, net, excluding other income.
- The scale was applied once and matches units_note.

## Examples
### Example 1
- **Situation:** A quarterly report shows three-month and nine-month columns for the current and prior year, headed "(in thousands)".
- **Method:** Four actual columns, each with its stated length; thousands applied once.
- **Result:** Four objects: months 9 and 3 for each year, period_type ytd and interim

### Example 2
- **Situation:** A statement of activities shows columns without donor restrictions, with donor restrictions and total; a row shows net assets released from restrictions.
- **Method:** Total column only; the released row nets to zero and is not added.
- **Result:** One object from the total column

### Example 3
- **Situation:** A management income statement shows month, year-to-date, budget and variance columns for the same period.
- **Method:** Month and year-to-date are actual; budget and variance are struck.
- **Result:** Two objects: months 1 and the stated year-to-date length
