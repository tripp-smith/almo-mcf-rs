use almo_mcf_core::scaling::{
    reduce_to_polynomial_capacities, reduce_to_polynomial_costs, unscale_flow,
};

#[test]
fn test_scaling_reductions() {
    let caps = vec![1_i64 << 30, 1_i64 << 29];
    let costs = vec![1_i64 << 20, -(1_i64 << 19)];
    let (c2, k_cost) = reduce_to_polynomial_costs(&costs, 1_i64 << 30, 1_i64 << 20);
    let (u2, k_cap) = reduce_to_polynomial_capacities(&caps, &[0, 0]);
    assert_eq!(c2.len(), costs.len());
    assert_eq!(u2.len(), caps.len());
    let restored = unscale_flow(&[1.0, 2.0], &[k_cost, k_cap]);
    assert_eq!(restored.len(), 2);
}
