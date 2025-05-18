import { useState } from 'react'
import axios from 'axios'

interface SearchResult {
  chunk_id: string
  text: string
  metadata: {
    source_file: string
    chunk_index: number
    strategy: string
  }
  distance: number
}

const SearchPage = () => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async () => {
    if (!query.trim()) return

    setIsLoading(true)
    setError(null)

    try {
      const response = await axios.post('/api/search', { query })
      setResults(response.data)
    } catch (err) {
      setError('搜索失败')
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-white shadow sm:rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg font-medium leading-6 text-gray-900">语义搜索</h3>
          <div className="mt-2 max-w-xl text-sm text-gray-500">
            <p>输入问题，系统将搜索相关文档片段</p>
          </div>
          <div className="mt-5">
            <div className="flex rounded-md shadow-sm">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                className="flex-1 min-w-0 block w-full px-3 py-2 rounded-md border border-gray-300 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                placeholder="输入搜索问题..."
              />
              <button
                onClick={handleSearch}
                disabled={isLoading}
                className="ml-3 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              >
                搜索
              </button>
            </div>
          </div>
        </div>
      </div>

      {isLoading && (
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
        </div>
      )}

      {error && (
        <div className="text-red-600 text-center">
          {error}
        </div>
      )}

      {!isLoading && !error && results.length > 0 && (
        <div className="bg-white shadow sm:rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg font-medium leading-6 text-gray-900 mb-4">搜索结果</h3>
            <div className="space-y-4">
              {results.map((result, index) => (
                <div
                  key={result.chunk_id}
                  className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50"
                >
                  <div className="flex justify-between items-start">
                    <div className="text-sm text-gray-500">
                      来源: {result.metadata.source_file}
                    </div>
                    <div className="text-sm text-gray-500">
                      相似度: {(1 - result.distance).toFixed(2)}
                    </div>
                  </div>
                  <div className="mt-2 text-gray-900">
                    {result.text}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {!isLoading && !error && results.length === 0 && query && (
        <div className="text-center text-gray-500">
          未找到相关结果
        </div>
      )}
    </div>
  )
}

export default SearchPage 