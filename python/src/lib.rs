use pyo3::prelude::*;

mod block_mcp;
mod elastic_net;
mod lasso;
mod mcp;
mod mtl_lasso;
mod multi_task_elastic_net;

#[pymodule]
fn _lib(_py: Python, m: &PyModule) -> PyResult<()> {
    // Lasso
    m.add_class::<lasso::LassoWrapper>()?;
    m.add_class::<mtl_lasso::MultiTaskLassoWrapper>()?;
    // ElasticNet
    m.add_class::<elastic_net::ElasticNetWrapper>()?;
    m.add_class::<multi_task_elastic_net::MultiTaskElasticNetWrapper>()?;
    // MCP
    m.add_class::<mcp::MCPWrapper>()?;
    m.add_class::<block_mcp::BlockMCPWrapper>()?;
    Ok(())
}
