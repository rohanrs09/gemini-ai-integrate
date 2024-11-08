"use client";

import { useState } from "react";
import { ClipboardIcon, DownloadIcon } from "@heroicons/react/solid"; // Ensure you're using the correct version of Heroicons

export default function Accordion({ data }) {
  const [activeIndex, setActiveIndex] = useState(null);
  const [copied, setCopied] = useState(false);

  const toggleAccordion = (index) => {
    setActiveIndex(activeIndex === index ? null : index);
  };

  const handleCopy = async (content) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true); // Show the feedback message
      setTimeout(() => setCopied(false), 3000); // Hide the message after 3 seconds
    } catch (err) {
      console.error("Failed to copy text: ", err);
    }
  };

  const handleDownload = (file) => {
    // Local file path (relative to the 'public' directory)
    const fileUrl = `/files/${file.url}`; // Adjusted for local files in the public folder
    
    const link = document.createElement("a");
    link.href = fileUrl; // URL for the file (local path)
    link.download = file.name; // Suggested name for the file download

    // Append the link to the body, trigger a click event, then remove the link
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="max-w-md mx-auto mt-8 space-y-4">
      {data.map((item, index) => (
        <div key={index} className="border border-gray-300 rounded-lg shadow-sm overflow-hidden">
          <button
            onClick={() => toggleAccordion(index)}
            className={`w-full text-left px-4 py-3 font-semibold text-gray-800 bg-gray-100 hover:bg-gray-200 transition-colors duration-200 ${
              activeIndex === index ? "bg-gray-200" : ""
            }`}
          >
            {item.title}
          </button>
          {activeIndex === index && (
            <div className="px-4 py-3 bg-white border-t border-gray-200 relative">
              <pre className="p-2 bg-gray-900 text-white rounded-lg overflow-x-auto relative">
                <code className="text-sm font-mono">{item.content}</code>
                <button
                  onClick={() => handleCopy(item.content)}
                  className="absolute top-2 right-2 text-gray-400 hover:text-white transition-colors duration-200"
                  aria-label="Copy to clipboard"
                >
                  <ClipboardIcon className="w-5 h-5" />
                </button>
              </pre>

              {/* Display download buttons for files */}
              <div className="mt-4">
                {item.files && item.files.length > 0 ? (
                  item.files.map((file, fileIndex) => (
                    <div key={fileIndex} className="flex items-center space-x-2">
                      <button
                        onClick={() => handleDownload(file)}
                        className="flex items-center space-x-2 text-gray-600 hover:text-gray-800 transition-colors duration-200"
                      >
                        <DownloadIcon className="w-5 h-5" />
                        <span>Download {file.name}</span>
                      </button>
                    </div>
                  ))
                ) : (
                  <p>No files available for download.</p>
                )}
              </div>

              {/* Show feedback message when copied */}
              {copied && (
                <div className="absolute top-2 left-1/2 transform -translate-x-1/2 bg-green-500 text-white text-sm rounded px-3 py-1">
                  Copied to clipboard!
                </div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
