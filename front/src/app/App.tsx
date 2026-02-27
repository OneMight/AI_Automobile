import { AdminHeader, Header, LoginedHeader } from "@/layouts";
import { useUser } from "@/lib/useUser";
import { isAdmin, isOwner } from "@/lib/utils";
import { Outlet } from "@tanstack/react-router";
import { Analytics } from "@vercel/analytics/react";
import { SpeedInsights } from "@vercel/speed-insights/react";
import { OwnerHeader } from "@/layouts";
function App() {
  const { user, isLoading } = useUser();
  const userIsOwner = isOwner(user?.role);
  const userIsAdmin = isAdmin(user?.role);
  return (
    <main className="w-full bg-main-app/80 min-h-screen flex flex-col items-center max-w-480 gap-15 relative">
      <Analytics />
      <SpeedInsights />
      {!isLoading && user ? (
        userIsOwner ? (
          <OwnerHeader />
        ) : userIsAdmin ? (
          <AdminHeader />
        ) : (
          <LoginedHeader />
        )
      ) : (
        <Header />
      )}
      <div className="w-full flex justify-center items-center">
        <Outlet />
      </div>
    </main>
  );
}

export default App;
