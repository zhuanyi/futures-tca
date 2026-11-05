# References

Each reference was resolved by DOI (Crossref) and its stated claim checked
against the abstract. Quoted wording was not checked against full paper text.

| # | Reference | Used for | DOI / landing URL |
|---|---|---|---|
| 1 | Huang, Lehalle & Rosenbaum (2015), "Simulating and Analyzing Order Book Data: The Queue-Reactive Model", *J. American Statistical Association* 110(509). Preprint arXiv:1312.0563. | Order-flow intensities that depend on book state; use of such a model as a simulator for transaction cost analysis. | https://doi.org/10.1080/01621459.2014.982278 |
| 2 | Cont, Stoikov & Talreja (2010), "A Stochastic Model for Order Book Dynamics", *Operations Research* 58(3). | Queue dynamics conditional on book state. The abstract does not name imbalance as a signal; do not attribute one to this paper as a headline claim. | https://doi.org/10.1287/opre.1090.0780 |
| 3 | Cont & de Larrard (2013), "Price Dynamics in a Markovian Limit Order Market", *SIAM J. Financial Mathematics* 4(1). | Markovian queueing book; price-move probabilities given book state. | https://doi.org/10.1137/110856605 |
| 4 | Dayri & Rosenbaum (2015), "Large Tick Assets: Implicit Spread and Optimal Tick Size", *Market Microstructure and Liquidity* 1(1). | Large-tick assets whose effective spread is almost always one tick. | https://doi.org/10.1142/s2382626615500033 |
| 5 | Moallemi & Yuan (2016), "A Model for Queue Position Valuation in a Limit Order Book", working paper (SSRN 2996221; initial version Dec 2016, revised Jun 2017). | Value of queue position. Unpublished; cite as a working paper. | https://doi.org/10.2139/ssrn.2996221 |
| 6 | Eisler, Bouchaud & Kockelkoren (2012), "The price impact of order book events: market orders, limit orders and cancellations", *Quantitative Finance* 12(9). arXiv:0904.0900. | Price impact of market orders, limit orders and cancellations. | https://doi.org/10.1080/14697688.2010.528444 |
| 7 | Stoikov (2018), "The micro-price: a high-frequency estimator of future prices", *Quantitative Finance* 18(12). SSRN 2970694 (2017 preprint). | Baseline only: the size-weighted mid. | https://doi.org/10.1080/14697688.2018.1489139 |

## Notes

- **Stoikov year:** the journal version is 2018; the SSRN preprint is 2017.
- **Weighted mid vs micro-price:** the benchmark `(b*Qa + a*Qb)/(Qa+Qb)` is the
  size-weighted mid (`wmid_arrival`), not Stoikov's micro-price. Stoikov defines
  the micro-price as the limit of a sequence of expected mid-prices and presents
  the weighted mid as the baseline it improves on. The micro-price estimator is
  an optional extension, not part of v1.
- **Roll parameters:** no roll references are cited; every roll parameter is a
  labelled assumption.
