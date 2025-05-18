import { useState } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'

interface GenerationResult {
  query: string
  answer: string
  context: Array<{
    text: string
    metadata: {
      source_file: string
      chunk_index: number
    }
  }>
  model: string
  usage: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

const GeneratePage = () => {
  const [query, setQuery] = useState('')
  const [result, setResult] = useState<GenerationResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [streamingAnswer, setStreamingAnswer] = useState('')

  const handleGenerate = async () => {
    if (!query.trim()) return

    setIsLoading(true)
    setError(null)
    setStreamingAnswer('')

    try {
      const response = await axios.post('/api/generate', { query })
      setResult(response.data)
    } catch (err) {
      setError('生成失败')
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleStreamGenerate = async () => {
    if (!query.trim()) return

    setIsLoading(true)
    setError(null)
    setStreamingAnswer('')

    try {
      const response = await axios.post('/api/generate/stream', { query }, {
        responseType: 'stream'
      })

      const reader = response.data.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const text = decoder.decode(value)
        setStreamingAnswer(prev => prev + text)
      }
    } catch (err) {
      setError('生成失败')
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-white shadow sm:rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg font-medium leading-6 text-gray-900">智能问答</h3>
          <div className="mt-2 max-w-xl text-sm text-gray-500">
            <p>输入问题，系统将基于文档内容生成回答</p>
          </div>
          <div className="mt-5">
            <div className="flex rounded-md shadow-sm">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                rows={3}
                className="flex-1 min-w-0 block w-full px-3 py-2 rounded-md border border-gray-300 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                placeholder="输入您的问题..."
              />
            </div>
            <div className="mt-3 flex space-x-3">
              <button
                onClick={handleGenerate}
                disabled={isLoading}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              >
                生成回答
              </button>
              <button
                onClick={handleStreamGenerate}
                disabled={isLoading}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              >
                流式生成
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

      {streamingAnswer && (
        <div className="bg-white shadow sm:rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg font-medium leading-6 text-gray-900 mb-4">生成结果</h3>
            <div className="prose max-w-none">
              <ReactMarkdown>{streamingAnswer}</ReactMarkdown>
            </div>
          </div>
        </div>
      )}

      {!isLoading && !error && result && !streamingAnswer && (
        <div className="bg-white shadow sm:rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg font-medium leading-6 text-gray-900 mb-4">生成结果</h3>
            <div className="prose max-w-none">
              <ReactMarkdown>{result.answer}</ReactMarkdown>
            </div>
            <div className="mt-6 border-t border-gray-200 pt-4">
              <h4 className="text-sm font-medium text-gray-900">参考文档</h4>
              <div className="mt-2 space-y-2">
                {result.context.map((item, index) => (
                  <div key={index} className="text-sm text-gray-500">
                    <div>来源: {item.metadata.source_file}</div>
                    <div className="mt-1">{item.text}</div>
                  </div>
                ))}
              </div>
            </div>
            <div className="mt-4 text-sm text-gray-500">
              <div>模型: {result.model}</div>
              <div>Token 使用: {result.usage.total_tokens}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default GeneratePage 