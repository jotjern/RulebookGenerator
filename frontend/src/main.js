const API_BASE = '/api'

async function updateDashboard() {
    try {
        const res = await fetch(`${API_BASE}/stl-conversions`)
        const data = await res.json()

        const conversions = data.conversions || []

        if (conversions.length === 0) {
            document.getElementById('conversions-list').innerHTML = '<p class="loading">No STL conversions available</p>'
            return
        }

        document.getElementById('conversions-list').innerHTML = conversions.map((conv, idx) => `
            <div class="conversion-item">
                <div class="rule-text">${idx + 1}. ${conv.rule_text}</div>
                <span class="category">${conv.category || 'uncategorized'}</span>

                <div class="stl-section">
                    <div class="label">STL</div>
                    <div class="formula">${conv.stl_formula}</div>
                </div>

                <div class="vs-section">
                    <div class="label">VS</div>
                    <div class="formula">${conv.value_sum}</div>
                </div>
            </div>
        `).join('')

    } catch (error) {
        console.error('Error:', error)
        document.getElementById('conversions-list').innerHTML = `<p class="loading">Error loading data</p>`
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        updateDashboard()
        setInterval(updateDashboard, 1000)
    })
} else {
    updateDashboard()
    setInterval(updateDashboard, 1000)
}
