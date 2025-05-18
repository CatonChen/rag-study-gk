import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { useState } from 'react'
import Navbar from './components/Navbar'
import DocumentPage from './pages/DocumentPage'
import SearchPage from './pages/SearchPage'
import GeneratePage from './pages/GeneratePage'

function App() {
  const [isLoading, setIsLoading] = useState(false)

  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<DocumentPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/generate" element={<GeneratePage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App 