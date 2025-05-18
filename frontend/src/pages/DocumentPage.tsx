import { useState } from 'react'
import axios from 'axios'

interface Document {
  filename: string
  file_size: number
  file_type: string
  created_time: number
  modified_time: number
}

const DocumentPage = () => {
  const [documents, setDocuments] = useState<Document[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files) return

    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i])
      }

      const response = await axios.post('/api/process', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setDocuments([...documents, ...response.data])
    } catch (err) {
      setError('文件上传失败')
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  const loadDocuments = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await axios.get('/api/list-docs')
      setDocuments(response.data)
    } catch (err) {
      setError('加载文档列表失败')
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-white shadow sm:rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg font-medium leading-6 text-gray-900">文档上传</h3>
          <div className="mt-2 max-w-xl text-sm text-gray-500">
            <p>支持上传 PDF、TXT、DOCX 等格式的文档</p>
          </div>
          <div className="mt-5">
            <input
              type="file"
              multiple
              onChange={handleFileUpload}
              className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-full file:border-0
                file:text-sm file:font-semibold
                file:bg-primary-50 file:text-primary-700
                hover:file:bg-primary-100"
            />
          </div>
        </div>
      </div>

      <div className="bg-white shadow sm:rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-medium leading-6 text-gray-900">文档列表</h3>
            <button
              onClick={loadDocuments}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
            >
              刷新
            </button>
          </div>
          
          {isLoading && (
            <div className="mt-4 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
            </div>
          )}

          {error && (
            <div className="mt-4 text-red-600">
              {error}
            </div>
          )}

          {!isLoading && !error && documents.length === 0 && (
            <div className="mt-4 text-center text-gray-500">
              暂无文档
            </div>
          )}

          {!isLoading && !error && documents.length > 0 && (
            <div className="mt-4">
              <div className="flex flex-col">
                <div className="-my-2 overflow-x-auto sm:-mx-6 lg:-mx-8">
                  <div className="py-2 align-middle inline-block min-w-full sm:px-6 lg:px-8">
                    <div className="shadow overflow-hidden border-b border-gray-200 sm:rounded-lg">
                      <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                          <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              文件名
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              大小
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              类型
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              修改时间
                            </th>
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                          {documents.map((doc, index) => (
                            <tr key={index}>
                              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                                {doc.filename}
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                {(doc.file_size / 1024).toFixed(2)} KB
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                {doc.file_type}
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                {new Date(doc.modified_time * 1000).toLocaleString()}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default DocumentPage 