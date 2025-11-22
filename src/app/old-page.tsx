
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      <header className="w-full py-4 bg-white shadow text-center">
        <span className="text-2xl font-semibold text-blue-700">TailAdmin Pro</span>
      </header>
      <main className="flex flex-1 flex-col items-center justify-center p-8">
        <h1 className="text-4xl font-bold mb-4">Welcome to TailAdmin Pro</h1>
        <p className="text-lg text-gray-600">This is your Next.js homepage. Start building your dashboard!</p>
        <div className="mt-8 flex gap-4">
          <Link href="/signin">
            <button className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition">Login</button>
          </Link>
          <Link href="/signup">
            <button className="px-6 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300 transition">Sign Up</button>
          </Link>
        </div>
      </main>
      <footer className="w-full py-4 bg-gray-100 text-center text-gray-500 text-sm">
        © {new Date().getFullYear()} TailAdmin Pro. All rights reserved.
      </footer>
    </div>
  );
}
