# Agent hard-v3 外部候选题库

状态：`questions_frozen_gold_pending`。当前只冻结题面，尚未制作独立 gold，也未运行 Agent。

题面 SHA-256：`a8650156ea0377b15955636d72208b77b9f582d1a5c57a52745eba2740595cb4`。

题量：96；公司：8；官方年报：16（2023/2024）；可回答：80；拒答或澄清：16。

任务分布：抽取 56、比较/核验 24、计算 16。以下清单不包含答案、证据页或目标文档 ID。

## 校准集

### 工商银行

- `v3_601398_y23_core` [extract/answer] 工商银行2023年净利润、资产总额、客户贷款及垫款总额和客户存款分别是多少？
- `v3_601398_y23_asset_quality` [extract/answer] 工商银行2023年不良贷款率、拨备覆盖率、资本充足率和成本收入比分别是多少？
- `v3_601398_y23_ecl_audit` [extract/answer] 工商银行2023年关于客户贷款及垫款预期信用损失的关键审计事项，风险点和审计应对分别是什么？
- `v3_601398_y24_core` [extract/answer] 工商银行2024年净利润、资产总额、客户贷款及垫款总额和客户存款分别是多少？
- `v3_601398_y24_interest` [extract/answer] 工商银行2024年净利息收入和净利息收益率如何变化，年报解释的主要原因是什么？
- `v3_601398_trend_core` [compare/answer] 比较工商银行2023年和2024年的净利润、资产总额、客户贷款及垫款总额和客户存款，分别判断增减。
- `v3_601398_trend_asset_quality` [compare/answer] 比较工商银行2023年和2024年的不良贷款率、拨备覆盖率、资本充足率和成本收入比。
- `v3_601398_calc_profit_growth` [calculate/answer] 用年报披露的净利润计算工商银行2024年相对2023年的增长率，保留两位小数。
- `v3_601398_calc_loan_deposit_gap` [calculate/answer] 分别计算工商银行2023年和2024年客户存款减客户贷款及垫款的差额，并判断差额是否扩大。
- `v3_601398_verify_quality` [verify/answer] 核验说法：工商银行2024年较2023年不良贷款率下降，同时资本充足率上升。请给出证据和结论。
- `v3_601398_future_unavailable` [extract/insufficient_evidence] 请给出工商银行2025年实际不良贷款率、拨备覆盖率和资本充足率，并引用当前语料中的年报页码。
- `v3_601398_ambiguous_scope` [clarify/clarify] 请比较这家银行两年的资产质量指标并解释变化，但先不要假定我指的是哪家银行或哪两个年度。

### 比亚迪

- `v3_002594_y23_core` [extract/answer] 比亚迪2023年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？
- `v3_002594_y23_products` [extract/answer] 比亚迪2023年汽车、汽车相关产品及其他产品与手机部件、组装及其他产品的营业收入和毛利率分别是多少？
- `v3_002594_y23_rd` [extract/answer] 比亚迪2023年研发投入金额、占营业收入比例、研发人员数量及占比分别是多少？
- `v3_002594_y24_core` [extract/answer] 比亚迪2024年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？
- `v3_002594_y24_products` [extract/answer] 比亚迪2024年两大产品类别的营业收入、同比变化和毛利率分别是多少？
- `v3_002594_trend_core` [compare/answer] 比较比亚迪2023年和2024年的营业收入、归母净利润和经营活动现金流量净额。
- `v3_002594_trend_rd` [compare/answer] 比较比亚迪2023年和2024年的研发投入金额、研发投入强度及研发人员数量。
- `v3_002594_calc_profit_margin` [calculate/answer] 用营业收入和归母净利润计算比亚迪2023年、2024年的归母净利率，并计算变化百分点。
- `v3_002594_calc_product_share` [calculate/answer] 计算比亚迪2024年汽车、汽车相关产品及其他产品收入占营业收入的比例，保留两位小数。
- `v3_002594_verify_growth` [verify/answer] 核验说法：比亚迪2024年营业收入和归母净利润均高于2023年，且净利润增速更快。
- `v3_002594_future_unavailable` [extract/insufficient_evidence] 请给出比亚迪2025年全年汽车业务收入和毛利率，并引用当前语料中的年报页码。
- `v3_002594_ambiguous_scope` [clarify/clarify] 请比较这家新能源汽车公司的盈利能力变化，但先确认公司名称、比较年度和所指利润口径。

## 开发集

### 中国平安

- `v3_601318_y23_core` [extract/answer] 中国平安2023年归母营运利润、归母净利润，以及按2023年末最新投资回报率和风险贴现率等假设计算的寿险及健康险新业务价值和内含价值分别是多少？
- `v3_601318_y23_segments` [extract/answer] 中国平安2023年寿险及健康险、财产保险、银行三项业务归属于母公司股东的营运利润分别是多少？
- `v3_601318_y23_audit` [extract/answer] 中国平安2023年保险合同负债计量相关关键审计事项的主要风险和审计应对是什么？
- `v3_601318_y24_core` [extract/answer] 中国平安2024年归母营运利润、归母净利润，以及按2024年末最新假设计算的寿险及健康险新业务价值和内含价值分别是多少？
- `v3_601318_y24_customer` [extract/answer] 中国平安2024年个人客户数、客均合同数以及持有四个及以上合同的客户留存率分别是多少？
- `v3_601318_trend_core` [compare/answer] 按中国平安2024年年报的可比口径（其中2023年归母营运利润为追溯调整后），比较2023年和2024年的归母营运利润、归母净利润及寿险新业务价值。
- `v3_601318_trend_customer` [compare/answer] 比较中国平安2023年和2024年的个人客户数、客均合同数以及持有集团内4个及以上合同客户的留存率。
- `v3_601318_calc_nbv_growth` [calculate/answer] 按中国平安2024年年报的可比口径，用披露值计算2024年寿险及健康险新业务价值相对2023年的增长率。
- `v3_601318_calc_segment_share` [calculate/answer] 计算中国平安2024年寿险及健康险业务归母营运利润占寿险及健康险、财产保险、银行三项核心业务归母营运利润合计的比例。
- `v3_601318_verify_nbv` [verify/answer] 按中国平安2024年年报的可比（调整后）口径核验说法：2024年寿险及健康险新业务价值较2023年增长，但内含价值没有增长。
- `v3_601318_future_unavailable` [extract/insufficient_evidence] 请给出中国平安2025年寿险及健康险新业务价值和归母营运利润，并引用当前语料中的年报页码。
- `v3_601318_ambiguous_scope` [clarify/clarify] 请分析这家保险公司两年的新业务价值变化，但先确认公司、年度以及新业务价值口径。

### 宁德时代

- `v3_300750_y23_core` [extract/answer] 宁德时代2023年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？
- `v3_300750_y23_products` [extract/answer] 宁德时代2023年动力电池系统、储能电池系统和电池材料及回收三类业务的收入和毛利率分别是多少？
- `v3_300750_y23_rd` [extract/answer] 宁德时代2023年研发费用、研发投入占营业收入比例和研发人员数量分别是多少？
- `v3_300750_y24_core` [extract/answer] 宁德时代2024年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？
- `v3_300750_y24_products` [extract/answer] 宁德时代2024年动力电池系统、储能电池系统和电池材料及回收业务的收入、同比变化和毛利率分别是多少？
- `v3_300750_trend_core` [compare/answer] 比较宁德时代2023年和2024年的营业收入、归母净利润和经营活动现金流量净额。
- `v3_300750_trend_products` [compare/answer] 比较宁德时代2023年和2024年动力电池系统与储能电池系统的收入和毛利率。
- `v3_300750_calc_storage_share` [calculate/answer] 计算宁德时代2024年储能电池系统收入占公司营业收入的比例，保留两位小数。
- `v3_300750_calc_net_margin` [calculate/answer] 计算宁德时代2023年和2024年的归母净利率，并给出变化百分点。
- `v3_300750_verify_divergence` [verify/answer] 核验说法：宁德时代2024年营业收入较2023年下降，但归母净利润上升。
- `v3_300750_future_unavailable` [extract/insufficient_evidence] 请给出宁德时代2025年动力电池系统收入和毛利率，并引用当前语料中的年报页码。
- `v3_300750_ambiguous_scope` [clarify/clarify] 请比较这家电池公司的业务结构变化，但先确认公司、年度和所指电池业务口径。

## 冻结测试集

### 万华化学

- `v3_600309_y23_core` [extract/answer] 万华化学2023年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？
- `v3_600309_y23_segments` [extract/answer] 万华化学2023年聚氨酯、石化和精细化学品及新材料三类业务的收入和毛利率分别是多少？
- `v3_600309_y23_rd` [extract/answer] 万华化学2023年研发投入合计、费用化金额、资本化金额及占营业收入比例分别是多少？
- `v3_600309_y24_core` [extract/answer] 万华化学2024年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？
- `v3_600309_y24_segments` [extract/answer] 万华化学2024年三类主要业务的收入、成本和毛利率分别是多少？
- `v3_600309_trend_core` [compare/answer] 比较万华化学2023年和2024年的营业收入、归母净利润和经营活动现金流量净额。
- `v3_600309_trend_segments` [compare/answer] 比较万华化学2023年和2024年聚氨酯、石化、精细化学品及新材料业务的毛利率。
- `v3_600309_calc_polyurethane_share` [calculate/answer] 计算万华化学2024年聚氨酯业务收入占营业收入的比例，保留两位小数。
- `v3_600309_calc_margin_spread` [calculate/answer] 计算万华化学2024年聚氨酯业务与石化业务毛利率的差值，并与2023年的差值比较。
- `v3_600309_verify_divergence` [verify/answer] 核验说法：万华化学2024年营业收入较2023年增长，但归母净利润下降。
- `v3_600309_future_unavailable` [extract/insufficient_evidence] 请给出万华化学2025年聚氨酯业务收入和毛利率，并引用当前语料中的年报页码。
- `v3_600309_ambiguous_scope` [clarify/clarify] 请比较这家化工公司的分部毛利率变化，但先确认公司、比较年度和分部口径。

### 顺丰控股

- `v3_002352_y23_core` [extract/answer] 顺丰控股2023年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？
- `v3_002352_y23_businesses` [extract/answer] 顺丰控股2023年各主要物流分部的营业收入及同比变化分别是多少？
- `v3_002352_y23_audit` [extract/answer] 顺丰控股2023年收入确认相关关键审计事项的主要风险和审计应对是什么？
- `v3_002352_y24_core` [extract/answer] 顺丰控股2024年营业收入、归母净利润、息税折旧摊销前利润和净资产收益率分别是多少？
- `v3_002352_y24_network` [extract/answer] 顺丰控股2024年国际快递、货代及供应链业务覆盖国家及地区数、全球运营管理干支线货车数、运营全货机数、全球累计运营航空线路数、铁路普列线路数和海运线路数分别是多少？请保留信息图中的大于号下限口径。
- `v3_002352_trend_core` [compare/answer] 比较顺丰控股2023年和2024年的营业收入、归母净利润和经营活动现金流量净额。
- `v3_002352_trend_businesses` [compare/answer] 比较顺丰控股2023年和2024年时效快递、经济快递、快运及冷运医药业务的收入变化。
- `v3_002352_calc_profit_margin` [calculate/answer] 计算顺丰控股2023年和2024年的归母净利率，并给出变化百分点。
- `v3_002352_calc_vehicle_rail_ratio` [calculate/answer] 根据顺丰控股2024年运力信息图披露的下限，计算全球运营管理干支线货车数量与铁路普列线路数量之比的下限，保留两位小数并保留大于号。
- `v3_002352_verify_summary` [verify/answer] 核验说法：顺丰控股2024年营业收入上升、归母净利润上升，但总资产下降。
- `v3_002352_future_unavailable` [extract/insufficient_evidence] 请给出顺丰控股2025年全球运营全货机数量和全年营业收入，并引用当前语料中的年报页码。
- `v3_002352_ambiguous_scope` [clarify/clarify] 请比较这家物流公司的网络能力变化，但先确认公司、年度和要比较的运输方式。

### 迈瑞医疗

- `v3_300760_y23_core` [extract/answer] 迈瑞医疗2023年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？
- `v3_300760_y23_products` [extract/answer] 迈瑞医疗2023年生命信息与支持、体外诊断和医学影像三类产品的收入和毛利率分别是多少？
- `v3_300760_y23_rd` [extract/answer] 迈瑞医疗2023年研发投入金额、占营业收入比例和研发人员数量分别是多少？
- `v3_300760_y24_core` [extract/answer] 迈瑞医疗2024年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？
- `v3_300760_y24_products` [extract/answer] 迈瑞医疗2024年三大产品线的收入、同比变化和毛利率分别是多少？
- `v3_300760_trend_core` [compare/answer] 比较迈瑞医疗2023年和2024年的营业收入、归母净利润和经营活动现金流量净额。
- `v3_300760_trend_products` [compare/answer] 按迈瑞医疗2024年年报追溯调整后的可比口径，比较2023年和2024年生命信息与支持、体外诊断、医学影像三大产品线的收入与毛利率。
- `v3_300760_calc_ivd_share` [calculate/answer] 计算迈瑞医疗2024年体外诊断产品收入占营业收入的比例，保留两位小数。
- `v3_300760_calc_margin_spread` [calculate/answer] 按迈瑞医疗2024年年报追溯调整后的可比口径，计算2024年三大产品线最高与最低毛利率之差，并与2023年的差值比较。
- `v3_300760_verify_growth` [verify/answer] 核验说法：迈瑞医疗2024年营业收入和归母净利润均高于2023年。
- `v3_300760_future_unavailable` [extract/insufficient_evidence] 请给出迈瑞医疗2025年体外诊断产品收入和毛利率，并引用当前语料中的年报页码。
- `v3_300760_ambiguous_scope` [clarify/clarify] 请比较这家医疗器械公司的产品结构变化，但先确认公司、年度和产品分类口径。

### 中兴通讯

- `v3_000063_y23_core` [extract/answer] 中兴通讯2023年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？
- `v3_000063_y23_markets` [extract/answer] 中兴通讯2023年运营商网络、政企业务和消费者业务的营业收入及毛利率分别是多少？
- `v3_000063_y23_rd` [extract/answer] 中兴通讯2023年研发投入金额、占营业收入比例和研发人员占比分别是多少？
- `v3_000063_y24_core` [extract/answer] 中兴通讯2024年营业收入、归母净利润、扣非归母净利润和经营活动现金流量净额分别是多少？
- `v3_000063_y24_markets` [extract/answer] 中兴通讯2024年三大业务市场的营业收入、同比变化和毛利率分别是多少？
- `v3_000063_trend_core` [compare/answer] 比较中兴通讯2023年和2024年的营业收入、归母净利润和经营活动现金流量净额。
- `v3_000063_trend_markets` [compare/answer] 按中兴通讯2024年年报追溯调整后的可比口径，比较2023年和2024年运营商网络、政企业务和消费者业务的收入与毛利率。
- `v3_000063_calc_enterprise_share` [calculate/answer] 计算中兴通讯2024年政企业务收入占营业收入的比例，保留两位小数。
- `v3_000063_calc_rd_change` [calculate/answer] 计算中兴通讯2024年研发投入相对2023年的增长率，并比较研发投入强度变化。
- `v3_000063_verify_divergence` [verify/answer] 核验说法：中兴通讯2024年营业收入较2023年增长，但归母净利润下降。
- `v3_000063_future_unavailable` [extract/insufficient_evidence] 请给出中兴通讯2025年政企业务收入和毛利率，并引用当前语料中的年报页码。
- `v3_000063_ambiguous_scope` [clarify/clarify] 请比较这家通信设备公司的三大市场表现，但先确认公司、年度和市场分类口径。
