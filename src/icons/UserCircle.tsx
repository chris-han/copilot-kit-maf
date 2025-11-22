import { UserCircle as LucideUserCircle, LucideProps } from 'lucide-react';

const UserCircle = ({ className, ...props }: LucideProps) => {
  return <LucideUserCircle className={className} {...props} />;
};

export default UserCircle;